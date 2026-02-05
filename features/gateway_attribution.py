from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.config.constants import (
    GATEWAY_PRODUCT_KEYWORDS,
    GATEWAY_SESSION_KEYWORDS,
    MENTORSHIP_KEYWORDS,
)


def _has_keyword(value: object, keywords: list[str]) -> bool:
    text = str(value).lower() if value is not None else ""
    return any(k in text for k in keywords)


def classify_gateway_sessions(live_sessions: pd.DataFrame) -> pd.DataFrame:
    sessions = live_sessions.copy()
    if "id" in sessions.columns and "liveSessionId" not in sessions.columns:
        sessions = sessions.rename(columns={"id": "liveSessionId"})
    sessions["sessionTitle"] = sessions.get("title")
    sessions["is_gateway_session"] = sessions["sessionTitle"].apply(lambda x: _has_keyword(x, GATEWAY_SESSION_KEYWORDS))
    return sessions[["liveSessionId", "sessionTitle", "scheduledAt", "createdById", "is_gateway_session"]]


def classify_products(
    products: pd.DataFrame,
    gateway_quantile: float,
    mentorship_quantile: float,
) -> pd.DataFrame:
    prod = products.copy()
    prod["productTitle"] = prod.get("title")

    price = prod.get("price")
    price_low = float(price.quantile(gateway_quantile)) if price is not None and price.notna().any() else np.nan
    price_high = float(price.quantile(mentorship_quantile)) if price is not None and price.notna().any() else np.nan

    prod["is_gateway_product"] = prod["productTitle"].apply(lambda x: _has_keyword(x, GATEWAY_PRODUCT_KEYWORDS))
    if not np.isnan(price_low):
        prod["is_gateway_product"] = prod["is_gateway_product"] | (prod["price"] <= price_low)

    prod["is_mentorship_product"] = prod["productTitle"].apply(lambda x: _has_keyword(x, MENTORSHIP_KEYWORDS))
    if not np.isnan(price_high):
        prod["is_mentorship_product"] = prod["is_mentorship_product"] | (prod["price"] >= price_high)

    return prod[["id", "productTitle", "price", "discountPrice", "is_gateway_product", "is_mentorship_product"]].rename(columns={"id": "productId"})


def build_gateway_attribution(
    live_sessions: pd.DataFrame,
    live_session_attendance: pd.DataFrame,
    live_session_assigned: pd.DataFrame,
    payments: pd.DataFrame,
    products: pd.DataFrame,
    product_accesses: pd.DataFrame,
    gateway_quantile: float,
    mentorship_quantile: float,
) -> dict[str, pd.DataFrame]:
    # Classifiers
    sessions = classify_gateway_sessions(live_sessions)
    prod_flags = classify_products(products, gateway_quantile, mentorship_quantile)

    # Gateway session touch
    attendance = live_session_attendance.rename(columns={"studentId": "userId"}).copy()
    attendance = attendance.merge(sessions, on="liveSessionId", how="left")
    gateway_attendance = attendance[attendance["is_gateway_session"] == True].copy()
    gateway_attendance = gateway_attendance.dropna(subset=["attendedAt"])
    gateway_attendance = gateway_attendance.sort_values("attendedAt")

    first_session_touch = gateway_attendance.groupby("userId").first().reset_index()
    first_session_touch = first_session_touch[
        ["userId", "attendedAt", "liveSessionId", "sessionTitle"]
    ].rename(
        columns={
            "attendedAt": "gateway_session_time",
            "liveSessionId": "gateway_session_id",
        }
    )

    # Gateway product touch
    payments_succ = payments[payments["status"] == "succeeded"].copy()
    payments_succ["paidAt"] = payments_succ["paidAt"].fillna(payments_succ["createdAt"])
    payments_succ = payments_succ.merge(prod_flags, on="productId", how="left")
    gateway_payments = payments_succ[payments_succ["is_gateway_product"] == True].copy()
    gateway_payments = gateway_payments.dropna(subset=["paidAt"])
    gateway_payments = gateway_payments.sort_values("paidAt")

    first_product_touch = gateway_payments.groupby("userId").first().reset_index()
    first_product_touch = first_product_touch[
        ["userId", "paidAt", "productId", "productTitle"]
    ].rename(
        columns={
            "paidAt": "gateway_product_time",
            "productId": "gateway_product_id",
        }
    )

    # Merge to get first gateway touch (session or product)
    touches = first_session_touch.merge(first_product_touch, on="userId", how="outer")
    touches["first_touch_time"] = touches[["gateway_session_time", "gateway_product_time"]].min(axis=1)
    touches["first_touch_type"] = np.where(
        touches["gateway_session_time"].notna()
        & ((touches["gateway_product_time"].isna()) | (touches["gateway_session_time"] <= touches["gateway_product_time"])),
        "session",
        "product",
    )
    touches["gateway_asset_name"] = np.where(
        touches["first_touch_type"] == "session",
        touches["sessionTitle"],
        touches["productTitle"],
    )

    touches = touches.dropna(subset=["first_touch_time"])

    # Earliest payments + mentorship payments + access
    first_payment = payments_succ.groupby("userId")["paidAt"].min().reset_index().rename(columns={"paidAt": "first_payment_time"})
    mentorship_payments = payments_succ[payments_succ["is_mentorship_product"] == True].copy()
    first_mentor_payment = mentorship_payments.groupby("userId")["paidAt"].min().reset_index().rename(columns={"paidAt": "first_mentorship_time"})

    accesses = product_accesses.copy()
    accesses["access_time"] = accesses["startDate"].fillna(accesses["createdAt"])
    first_access = accesses.groupby("userId")["access_time"].min().reset_index().rename(columns={"access_time": "first_access_time"})

    touches = touches.merge(first_payment, on="userId", how="left")
    touches = touches.merge(first_mentor_payment, on="userId", how="left")
    touches = touches.merge(first_access, on="userId", how="left")

    # Newcomer definitions (A/B/C) + strict intersection
    touches["is_newcomer_A"] = touches["first_payment_time"].isna() | (touches["first_payment_time"] >= touches["first_touch_time"])
    touches["is_newcomer_B"] = touches["first_mentorship_time"].isna() | (touches["first_mentorship_time"] >= touches["first_touch_time"])
    touches["is_newcomer_C"] = touches["first_access_time"].isna() | (touches["first_access_time"] >= touches["first_touch_time"])
    touches["is_newcomer_strict"] = touches["is_newcomer_A"] & touches["is_newcomer_B"] & touches["is_newcomer_C"]

    # Conversion after gateway touch
    merged = touches[["userId", "first_touch_time"]].merge(
        payments_succ[["userId", "paidAt", "productId", "is_mentorship_product"]],
        on="userId",
        how="left",
    )
    merged = merged[merged["paidAt"] > merged["first_touch_time"]]
    first_any_after = merged.groupby("userId")["paidAt"].min().reset_index().rename(columns={"paidAt": "first_any_paid_after"})
    first_mentor_after = merged[merged["is_mentorship_product"] == True].groupby("userId")["paidAt"].min().reset_index().rename(
        columns={"paidAt": "first_mentorship_paid_after"}
    )

    touches = touches.merge(first_any_after, on="userId", how="left")
    touches = touches.merge(first_mentor_after, on="userId", how="left")
    touches["days_to_any_paid"] = (touches["first_any_paid_after"] - touches["first_touch_time"]).dt.days
    touches["days_to_mentorship_paid"] = (touches["first_mentorship_paid_after"] - touches["first_touch_time"]).dt.days

    # Conversion summary
    windows = [1, 3, 7, 14, 30]
    cohort_defs = {
        "newcomer_A": "is_newcomer_A",
        "newcomer_B": "is_newcomer_B",
        "newcomer_C": "is_newcomer_C",
        "newcomer_strict": "is_newcomer_strict",
    }

    summary_rows = []
    for cohort_name, col in cohort_defs.items():
        cohort = touches[touches[col] == True]
        size = int(cohort["userId"].nunique())
        row = {"cohort": cohort_name, "cohort_size": size}
        for w in windows:
            any_count = int((cohort["days_to_any_paid"].notna() & (cohort["days_to_any_paid"] <= w)).sum())
            ment_count = int((cohort["days_to_mentorship_paid"].notna() & (cohort["days_to_mentorship_paid"] <= w)).sum())
            row[f"any_paid_{w}d"] = any_count
            row[f"any_paid_{w}d_rate"] = any_count / size if size else np.nan
            row[f"mentorship_paid_{w}d"] = ment_count
            row[f"mentorship_paid_{w}d_rate"] = ment_count / size if size else np.nan
        row["median_days_any_paid"] = float(cohort["days_to_any_paid"].median()) if size else np.nan
        row["median_days_mentorship_paid"] = float(cohort["days_to_mentorship_paid"].median()) if size else np.nan
        summary_rows.append(row)

    conversion_summary = pd.DataFrame(summary_rows)

    # Conversion curve (mentorship) for strict cohort
    curve_rows = []
    for cohort_name, col in cohort_defs.items():
        cohort = touches[touches[col] == True]
        size = int(cohort["userId"].nunique())
        if size == 0:
            continue
        for day in range(0, 31):
            rate = float((cohort["days_to_mentorship_paid"].notna() & (cohort["days_to_mentorship_paid"] <= day)).sum()) / size
            curve_rows.append({"cohort": cohort_name, "day": day, "mentorship_conversion_rate": rate})

    conversion_curve = pd.DataFrame(curve_rows)

    # Asset conversion (strict newcomers)
    asset_rows = []
    strict = touches[touches["is_newcomer_strict"] == True]
    if not strict.empty:
        for asset_type, name_col in [("session", "sessionTitle"), ("product", "productTitle")]:
            subset = strict[strict["first_touch_type"] == asset_type].copy()
            if subset.empty:
                continue
            group = subset.groupby(name_col, dropna=False)
            for asset_name, grp in group:
                touches_count = int(grp["userId"].nunique())
                conv_14 = int((grp["days_to_mentorship_paid"].notna() & (grp["days_to_mentorship_paid"] <= 14)).sum())
                asset_rows.append(
                    {
                        "asset_type": asset_type,
                        "asset_name": asset_name,
                        "touches": touches_count,
                        "mentorship_converted_14d": conv_14,
                        "conversion_rate_14d": conv_14 / touches_count if touches_count else np.nan,
                    }
                )

    asset_conversion = pd.DataFrame(asset_rows)

    # Session mix (newcomers vs existing mentorship)
    assigned = live_session_assigned.groupby("liveSessionId")["userId"].nunique().reset_index().rename(columns={"userId": "assignedCount"})
    attended = live_session_attendance.groupby("liveSessionId")["studentId"].nunique().reset_index().rename(columns={"studentId": "attendedCount"})
    session_mix = sessions.merge(assigned, on="liveSessionId", how="left").merge(attended, on="liveSessionId", how="left")
    if "assignedCount" not in session_mix.columns:
        session_mix["assignedCount"] = 0
    if "attendedCount" not in session_mix.columns:
        session_mix["attendedCount"] = 0
    session_mix["assignedCount"] = session_mix["assignedCount"].fillna(0)
    session_mix["attendedCount"] = session_mix["attendedCount"].fillna(0)
    session_mix["joinRate"] = session_mix["attendedCount"] / session_mix["assignedCount"].replace(0, np.nan)

    # New face rate (first attendance ever)
    first_att = live_session_attendance.groupby("studentId")["attendedAt"].min().reset_index().rename(columns={"studentId": "userId", "attendedAt": "first_attendedAt"})
    attendance_nf = live_session_attendance.rename(columns={"studentId": "userId"}).merge(first_att, on="userId", how="left")
    attendance_nf["isNewFace"] = attendance_nf["attendedAt"] == attendance_nf["first_attendedAt"]
    new_faces = attendance_nf.groupby("liveSessionId")["isNewFace"].sum().reset_index().rename(columns={"isNewFace": "newFaces"})
    session_mix = session_mix.merge(new_faces, on="liveSessionId", how="left")
    if "newFaces" not in session_mix.columns:
        session_mix["newFaces"] = 0
    session_mix["newFaces"] = session_mix["newFaces"].fillna(0)
    session_mix["newFaceRate"] = session_mix["newFaces"] / session_mix["attendedCount"].replace(0, np.nan)

    # Existing mentorship vs newcomer strict by session (gateway sessions only)
    session_att = attendance.rename(columns={"studentId": "userId"}).copy()
    session_att = session_att.merge(first_payment, on="userId", how="left")
    session_att = session_att.merge(first_mentor_payment, on="userId", how="left")
    session_att = session_att.merge(first_access, on="userId", how="left")
    session_att["is_existing_mentor"] = session_att["first_mentorship_time"].notna() & (session_att["first_mentorship_time"] < session_att["attendedAt"])
    session_att["is_newcomer_strict"] = (
        (session_att["first_payment_time"].isna() | (session_att["first_payment_time"] >= session_att["attendedAt"]))
        & (session_att["first_mentorship_time"].isna() | (session_att["first_mentorship_time"] >= session_att["attendedAt"]))
        & (session_att["first_access_time"].isna() | (session_att["first_access_time"] >= session_att["attendedAt"]))
    )
    session_att = session_att[session_att["is_gateway_session"] == True]

    if not session_att.empty:
        mix_counts = session_att.groupby("liveSessionId").agg(
            existing_mentor_count=("is_existing_mentor", "sum"),
            newcomer_strict_count=("is_newcomer_strict", "sum"),
            attendedCount=("userId", "nunique"),
        ).reset_index()
        mix_counts = mix_counts.rename(columns={"attendedCount": "attendedCount_calc"})
        mix_counts["existing_mentor_rate"] = mix_counts["existing_mentor_count"] / mix_counts["attendedCount_calc"].replace(0, np.nan)
        mix_counts["newcomer_strict_rate"] = mix_counts["newcomer_strict_count"] / mix_counts["attendedCount_calc"].replace(0, np.nan)
        session_mix = session_mix.merge(mix_counts, on="liveSessionId", how="left")
    else:
        session_mix["existing_mentor_rate"] = np.nan
        session_mix["newcomer_strict_rate"] = np.nan

    session_mix = session_mix[session_mix["is_gateway_session"] == True]

    # Final tables
    touches = touches[
        [
            "userId",
            "first_touch_time",
            "first_touch_type",
            "gateway_asset_name",
            "gateway_session_id",
            "sessionTitle",
            "gateway_product_id",
            "productTitle",
            "is_newcomer_A",
            "is_newcomer_B",
            "is_newcomer_C",
            "is_newcomer_strict",
            "days_to_any_paid",
            "days_to_mentorship_paid",
        ]
    ].rename(
        columns={
            "sessionTitle": "gateway_session_title",
            "productTitle": "gateway_product_title",
        }
    )

    session_mix = session_mix[
        [
            "liveSessionId",
            "sessionTitle",
            "scheduledAt",
            "assignedCount",
            "attendedCount",
            "joinRate",
            "newFaceRate",
            "existing_mentor_rate",
            "newcomer_strict_rate",
        ]
    ]

    return {
        "gateway_touch_attribution": touches,
        "gateway_conversion_summary": conversion_summary,
        "gateway_conversion_curve": conversion_curve,
        "gateway_asset_conversion": asset_conversion,
        "gateway_session_mix": session_mix,
    }
