from __future__ import annotations

import pandas as pd
import numpy as np

from analytics.io.loaders import month_start


def payment_status_by_month(payments: pd.DataFrame) -> pd.DataFrame:
    payments = payments.copy()
    payments["paymentMonth"] = month_start(payments["createdAt"])
    return payments.groupby(["paymentMonth", "status"], dropna=False)["id"].nunique().reset_index()


def revenue_by_month(payments: pd.DataFrame) -> pd.DataFrame:
    payments = payments[payments["status"] == "succeeded"].copy()
    payments["paymentMonth"] = month_start(payments["createdAt"])
    return payments.groupby("paymentMonth")["amount"].sum().reset_index()


def payment_delinquency(payments: pd.DataFrame, max_date: pd.Timestamp) -> pd.DataFrame:
    delinquent = payments[(payments["status"].isin(["pending", "not_paid"])) & (payments["dueDate"].notna())].copy()
    delinquent["isOverdue"] = delinquent["dueDate"] < max_date
    return delinquent.groupby("status")["id"].nunique().reset_index()


def paid_in_full_by_product(payments: pd.DataFrame) -> pd.DataFrame:
    payments = payments.copy()
    payments["status"] = payments["status"].fillna("unknown")
    grouped = payments.groupby(["userId", "productId"], dropna=False)
    summary = grouped.agg(
        succeeded_count=("status", lambda s: (s == "succeeded").sum()),
        any_non_succeeded=("status", lambda s: (s != "succeeded").any()),
        total_installments=("totalInstallments", "max"),
    ).reset_index()

    summary["paid_in_full"] = (~summary["any_non_succeeded"]) & (
        summary["total_installments"].isna() | (summary["succeeded_count"] >= summary["total_installments"])
    )

    return summary.groupby("productId").agg(
        users=("userId", "nunique"),
        paid_in_full_rate=("paid_in_full", "mean"),
    ).reset_index()


def _installment_tags(payments: pd.DataFrame, payment_commitments: pd.DataFrame, custom_products: pd.DataFrame) -> pd.DataFrame:
    payments_tag = payments[["userId", "productId", "totalInstallments"]].copy()
    payments_tag["is_installment"] = payments_tag["totalInstallments"].fillna(1) > 1

    commitments_tag = payment_commitments[["userId", "productId"]].copy()
    commitments_tag["is_installment"] = True

    custom_tag = custom_products[["userId", "productId", "paymentType"]].copy()
    custom_tag["is_installment"] = custom_tag["paymentType"].astype(str).str.lower().eq("split")

    tags = pd.concat([payments_tag[["userId", "productId", "is_installment"]], commitments_tag, custom_tag[["userId", "productId", "is_installment"]]], ignore_index=True)
    tags = tags.dropna(subset=["userId", "productId"])
    tags["is_installment"] = tags["is_installment"].fillna(False)

    return tags.groupby(["userId", "productId"])['is_installment'].max().reset_index()


def payment_plan_default_rate(payment_agreements: pd.DataFrame, payment_commitments: pd.DataFrame) -> pd.DataFrame:
    commitments = payment_commitments.copy()
    defaults = commitments.groupby("paymentAgreementId").agg(
        succeeded=("status", lambda s: (s == "succeeded").sum()),
        failed=("status", lambda s: (s != "succeeded").sum()),
    ).reset_index()

    agreements = payment_agreements.merge(defaults, left_on="id", right_on="paymentAgreementId", how="left")
    agreements["succeeded"] = agreements["succeeded"].fillna(0)
    agreements["failed"] = agreements["failed"].fillna(0)
    agreements["defaulted"] = (agreements["succeeded"] == 0) | (agreements["failed"] > 0)

    summary = agreements.groupby("reason")["defaulted"].mean().reset_index().rename(columns={"defaulted": "default_rate"})
    return summary


def commitment_vs_cash(payments: pd.DataFrame, payment_commitments: pd.DataFrame, custom_products: pd.DataFrame) -> pd.DataFrame:
    cash_collected = payments[payments["status"] == "succeeded"]["amount"].sum()
    commitment_collected = payment_commitments[payment_commitments["status"] == "succeeded"]["amount"].sum()

    contracted_value = custom_products["totalPrice"].sum() if "totalPrice" in custom_products.columns else 0.0

    outstanding = max(contracted_value - (cash_collected + commitment_collected), 0)

    return pd.DataFrame(
        {
            "stage": ["Contracted Value", "Cash Collected", "Outstanding"],
            "amount": [contracted_value, cash_collected + commitment_collected, outstanding],
        }
    )


def discount_hook_summary(products: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    prod = products[["id", "price", "discountPrice"]].rename(columns={"id": "productId"})
    merged = payments.merge(prod, on="productId", how="left")

    merged["is_discount"] = merged["amount"].round(2) == merged["discountPrice"].round(2)
    merged["is_full"] = merged["amount"].round(2) == merged["price"].round(2)

    return merged.groupby("productId").agg(
        discount_sales=("is_discount", "sum"),
        full_sales=("is_full", "sum"),
        total_sales=("id", "count"),
    ).reset_index()


def payment_plan_engagement(assignments_submissions: pd.DataFrame, payments: pd.DataFrame, payment_commitments: pd.DataFrame, custom_products: pd.DataFrame) -> pd.DataFrame:
    tags = _installment_tags(payments, payment_commitments, custom_products)

    submission_counts = assignments_submissions.groupby("studentId")["id"].nunique().reset_index().rename(columns={"studentId": "userId", "id": "submissionCount"})
    merged = tags.merge(submission_counts, on="userId", how="left")
    merged["submissionCount"] = merged["submissionCount"].fillna(0)

    return merged.groupby("is_installment").agg(
        users=("userId", "nunique"),
        avg_submissions=("submissionCount", "mean"),
    ).reset_index()


def investment_vs_engagement(assignments_submissions: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    payments = payments.copy()
    user_spend = payments[payments["status"] == "succeeded"].groupby("userId")["amount"].sum().reset_index()
    if user_spend.empty:
        return pd.DataFrame()

    threshold = user_spend["amount"].quantile(0.75)
    user_spend["investment_tier"] = np.where(user_spend["amount"] >= threshold, "high", "low")

    submission_counts = assignments_submissions.groupby("studentId")["id"].nunique().reset_index().rename(columns={"studentId": "userId", "id": "submissionCount"})
    merged = user_spend.merge(submission_counts, on="userId", how="left")
    merged["submissionCount"] = merged["submissionCount"].fillna(0)

    return merged.groupby("investment_tier").agg(
        users=("userId", "nunique"),
        avg_submissions=("submissionCount", "mean"),
        avg_spend=("amount", "mean"),
    ).reset_index()
