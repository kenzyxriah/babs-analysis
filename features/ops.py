from __future__ import annotations

import pandas as pd
import numpy as np


def ops_gap_report(enrollments: pd.DataFrame, payments: pd.DataFrame, login_history: pd.DataFrame, max_date: pd.Timestamp) -> pd.DataFrame:
    succeeded = payments[payments["status"] == "succeeded"][["userId", "productId"]].drop_duplicates()
    enroll_pairs = enrollments[["userId", "productId"]].drop_duplicates()

    paid_not_assigned = succeeded.merge(enroll_pairs, on=["userId", "productId"], how="left", indicator=True)
    paid_not_assigned = paid_not_assigned[paid_not_assigned["_merge"] == "left_only"]

    assigned_not_paid = enroll_pairs.merge(succeeded, on=["userId", "productId"], how="left", indicator=True)
    assigned_not_paid = assigned_not_paid[assigned_not_paid["_merge"] == "left_only"]

    login_success = login_history[login_history["status"] == "success"]
    last_login = login_success.groupby("userId")["timestamp"].max().reset_index()
    cutoff = max_date - pd.Timedelta(days=30)
    inactive_users = last_login[last_login["timestamp"] < cutoff]["userId"]
    access_no_login = enrollments[enrollments["userId"].isin(inactive_users)]["userId"].nunique()

    rows = [
        {"gapType": "paid_not_assigned", "userCount": paid_not_assigned["userId"].nunique(), "notes": "Succeeded payment without enrollment record"},
        {"gapType": "assigned_not_paid", "userCount": assigned_not_paid["userId"].nunique(), "notes": "Enrollment without a succeeded payment"},
        {"gapType": "active_access_no_recent_login", "userCount": access_no_login, "notes": "Enrollment with no login in last 30 days"},
    ]

    return pd.DataFrame(rows)


def exception_duration_summary(payment_exceptions: pd.DataFrame) -> pd.DataFrame:
    exceptions = payment_exceptions.copy()
    exceptions["reason"] = exceptions["reason"].fillna("").astype(str).str.strip().replace({"": "Unknown"})
    exceptions["durationDays"] = (exceptions["endDate"] - exceptions["startDate"]).dt.days
    return exceptions.groupby("reason")["durationDays"].agg(["count", "mean", "median"]).reset_index()


def exception_timeline(payment_exceptions: pd.DataFrame) -> pd.DataFrame:
    exceptions = payment_exceptions.copy()
    exceptions = exceptions.dropna(subset=["startDate", "endDate"])
    exceptions = exceptions.sort_values("startDate")
    exceptions["durationDays"] = (exceptions["endDate"] - exceptions["startDate"]).dt.days
    return exceptions[["id", "userId", "reason", "startDate", "endDate", "durationDays"]]


def sales_lag(form_submissions: pd.DataFrame, users: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    leads = form_submissions.copy()
    if "id" not in leads.columns and "userId" in leads.columns:
        leads = leads.rename(columns={"userId": "id"})
    if "id" not in leads.columns:
        leads = leads.merge(users[["id", "email"]], left_on="email", right_on="email", how="left")
    paid = payments[payments["status"] == "succeeded"].sort_values("paidAt")

    first_paid = paid.groupby("userId").first().reset_index()[["userId", "paidAt", "amount"]]
    merged = leads.merge(first_paid, left_on="id", right_on="userId", how="left")

    merged["salesLagDays"] = (merged["paidAt"] - merged["submittedAt"]).dt.days
    return merged[["submissionId", "email", "submittedAt", "paidAt", "amount", "salesLagDays"]]


def golden_layer_correlations(
    leads_llm: pd.DataFrame,
    engagement: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    spend = payments[payments["status"] == "succeeded"].groupby("userId")["amount"].sum().reset_index()
    merged = leads_llm.merge(spend, on="userId", how="left")
    merged = merged.merge(engagement, on="userId", how="left")

    numeric_cols = merged.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    corr = merged[numeric_cols].corr().reset_index().rename(columns={"index": "metric"})
    return corr
