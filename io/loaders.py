from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_pkl(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.pkl"
    return pd.read_pickle(path)


def to_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def month_start(series: pd.Series) -> pd.Series:
    return series.dt.to_period("M").dt.to_timestamp()


def load_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "users",
        "roles",
        "courses",
        "modules",
        "assignments",
        "assignment_submissions",
        "assignment_user_agreements",
        "module_assigned_users",
        "live_sessions",
        "live_session_assigned_students",
        "live_session_attendance",
        "products",
        "product_assets",
        "product_accesses",
        "payments",
        "payment_commitments",
        "payment_agreements",
        "payment_exceptions",
        "custom_products",
        "form",
        "form_submission",
        "login_history",
        "catalogues",
        "catalogue_categories",
        "categories",
        "tags",
        "course_tags",
        "product_tags",
        "programs",
        "program_courses",
        "product_programs",
        "program_tags",
        "user_program_selections",
    ]

    data = {name: load_pkl(data_dir, name) for name in names}

    data["users"] = to_datetime(data["users"], ["createdAt", "updatedAt", "lastActive"])
    data["payments"] = to_datetime(data["payments"], ["createdAt", "updatedAt", "paidAt", "dueDate"])
    data["payment_commitments"] = to_datetime(data["payment_commitments"], ["createdAt", "paidAt", "updatedAt"])
    data["payment_agreements"] = to_datetime(data["payment_agreements"], ["createdAt", "updatedAt", "signedAt"])
    data["product_accesses"] = to_datetime(data["product_accesses"], ["createdAt", "startDate", "endDate"])
    data["live_sessions"] = to_datetime(data["live_sessions"], ["scheduledAt", "createdAt", "updatedAt"])
    data["live_session_attendance"] = to_datetime(data["live_session_attendance"], ["attendedAt", "createdAt", "updatedAt"])
    data["assignment_submissions"] = to_datetime(data["assignment_submissions"], ["submittedAt", "gradedAt", "createdAt", "updatedAt"])
    data["assignments"] = to_datetime(data["assignments"], ["startDate", "dueDate", "publishedAt", "createdAt", "updatedAt"])
    data["form_submission"] = to_datetime(data["form_submission"], ["submittedAt", "deletedAt"])
    data["payment_exceptions"] = to_datetime(data["payment_exceptions"], ["startDate", "endDate", "createdAt", "updatedAt"])
    data["catalogues"] = to_datetime(data["catalogues"], ["createdAt", "updatedAt", "deletedAt"])
    data["catalogue_categories"] = to_datetime(data["catalogue_categories"], ["createdAt", "updatedAt", "deletedAt"])
    data["categories"] = to_datetime(data["categories"], ["createdAt", "updatedAt"])
    data["tags"] = to_datetime(data["tags"], ["createdAt", "updatedAt"])
    data["course_tags"] = to_datetime(data["course_tags"], ["createdAt", "updatedAt", "deletedAt"])
    data["product_tags"] = to_datetime(data["product_tags"], ["createdAt", "updatedAt", "deletedAt"])
    data["program_tags"] = to_datetime(data["program_tags"], ["createdAt", "updatedAt", "deletedAt"])
    data["programs"] = to_datetime(data["programs"], ["createdAt", "updatedAt"])
    data["program_courses"] = to_datetime(data["program_courses"], ["createdAt", "updatedAt"])
    data["product_programs"] = to_datetime(data["product_programs"], ["createdAt", "updatedAt"])
    data["user_program_selections"] = to_datetime(data["user_program_selections"], ["createdAt", "updatedAt"])

    return data
