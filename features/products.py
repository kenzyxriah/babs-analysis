from __future__ import annotations

import pandas as pd
import numpy as np

from analytics.config.constants import GATEWAY_KEYWORDS, MENTORSHIP_KEYWORDS


def course_product_map(product_assets: pd.DataFrame) -> pd.DataFrame:
    return product_assets[["courseId", "productId"]].dropna().drop_duplicates()


def gateway_upgrade(
    payments: pd.DataFrame,
    products: pd.DataFrame,
    gateway_quantile: float,
    mentorship_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prod = products[["id", "title", "price"]].rename(columns={"id": "productId"})
    price_low = prod["price"].quantile(gateway_quantile)
    price_high = prod["price"].quantile(mentorship_quantile)

    def has_keyword(title: str, keywords: list[str]) -> bool:
        title_lower = str(title).lower()
        return any(k in title_lower for k in keywords)

    prod["is_gateway"] = prod.apply(lambda r: has_keyword(r["title"], GATEWAY_KEYWORDS) and r["price"] <= price_low, axis=1)
    prod["is_mentorship"] = prod.apply(lambda r: has_keyword(r["title"], MENTORSHIP_KEYWORDS) or r["price"] >= price_high, axis=1)

    pay = payments[payments["status"] == "succeeded"].merge(prod, on="productId", how="left")
    pay = pay.sort_values("paidAt")

    first_purchase = pay.groupby("userId").head(1)
    later_purchase = pay.groupby("userId").tail(1).rename(columns={"is_mentorship": "is_mentorship_later"})

    merged = first_purchase[["userId", "productId", "paidAt", "is_gateway"]].merge(
        later_purchase[["userId", "productId", "paidAt", "is_mentorship_later"]],
        on="userId",
        suffixes=("_first", "_later"),
        how="left",
    )

    merged["upgraded_to_mentorship"] = merged["is_gateway"] & merged["is_mentorship_later"]
    merged["upgrade_days"] = (merged["paidAt_later"] - merged["paidAt_first"]).dt.days

    summary = merged.groupby("is_gateway")["upgraded_to_mentorship"].mean().reset_index().rename(columns={"is_gateway": "gateway_flag", "upgraded_to_mentorship": "upgrade_rate"})
    timeline = merged[
        [
            "userId",
            "productId_first",
            "productId_later",
            "is_gateway",
            "is_mentorship_later",
            "upgrade_days",
            "upgraded_to_mentorship",
        ]
    ]

    return summary, timeline


def bundle_utilization(products: pd.DataFrame, product_assets: pd.DataFrame, module_assigned_users: pd.DataFrame, modules: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    course_map = course_product_map(product_assets)
    bundle_sizes = course_map.groupby("productId")["courseId"].nunique().reset_index().rename(columns={"courseId": "courseCount"})
    bundles = bundle_sizes[bundle_sizes["courseCount"] > 1]

    paid_users = payments[payments["status"] == "succeeded"][["userId", "productId"]].drop_duplicates()
    paid_bundles = paid_users.merge(bundles, on="productId", how="inner")

    module_course = modules[["id", "courseId"]].rename(columns={"id": "moduleId"})
    module_users = module_assigned_users.merge(module_course, on="moduleId", how="left")

    usage = paid_bundles.merge(course_map, on="productId", how="left")
    usage = usage.merge(module_users, on=["userId", "courseId"], how="left", indicator=True)
    usage["entered_course"] = usage["_merge"] == "both"

    utilization = usage.groupby(["userId", "productId", "courseCount"]).agg(
        courses_entered=("entered_course", "sum")
    ).reset_index()
    utilization["utilization_rate"] = utilization["courses_entered"] / utilization["courseCount"].replace(0, np.nan)

    return utilization


def product_performance_matrix(payments: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    revenue = payments[payments["status"] == "succeeded"].groupby("productId").agg(
        units=("id", "count"),
        revenue=("amount", "sum"),
    ).reset_index()

    merged = revenue.merge(products[["id", "title"]], left_on="productId", right_on="id", how="left")
    merged = merged.rename(columns={"title": "productTitle"})
    return merged.drop(columns=["id"], errors="ignore")


def product_revenue_pareto(payments: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    revenue = payments[payments["status"] == "succeeded"].groupby("productId").agg(
        units=("id", "count"),
        revenue=("amount", "sum"),
    ).reset_index().sort_values("revenue", ascending=False)
    revenue["cumulative_revenue"] = revenue["revenue"].cumsum()
    total = revenue["revenue"].sum()
    revenue["cumulative_share"] = revenue["cumulative_revenue"] / total if total else 0

    merged = revenue.merge(products[["id", "title"]], left_on="productId", right_on="id", how="left")
    merged = merged.rename(columns={"title": "productTitle"})
    return merged.drop(columns=["id"], errors="ignore")


def module_saturation(modules: pd.DataFrame, module_assigned_users: pd.DataFrame) -> pd.DataFrame:
    counts = module_assigned_users.groupby("moduleId")["userId"].nunique().reset_index().rename(columns={"userId": "assignedUsers"})
    merged = modules.merge(counts, left_on="id", right_on="moduleId", how="left")
    merged["assignedUsers"] = merged["assignedUsers"].fillna(0)

    if "maxParticipants" in merged.columns:
        merged["capacity"] = merged["maxParticipants"].fillna(0)
        merged["saturation"] = merged["assignedUsers"] / merged["capacity"].replace(0, np.nan)

    return merged[["id", "title", "assignedUsers", "maxParticipants", "saturation"]]
