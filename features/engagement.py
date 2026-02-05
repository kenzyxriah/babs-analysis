from __future__ import annotations

import pandas as pd
import numpy as np

from analytics.io.loaders import to_datetime


def build_assignment_completion(assignments: pd.DataFrame, assignment_submissions: pd.DataFrame, modules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = assignments.merge(modules[["id", "courseId"]], left_on="moduleId", right_on="id", how="left", suffixes=("", "_module"))

    assignments_per_course = assignments.groupby("courseId", dropna=False)["id"].nunique().reset_index()
    assignments_per_course = assignments_per_course.rename(columns={"id": "totalAssignments"})

    submissions = assignment_submissions.merge(
        assignments[["id", "moduleId", "courseId"]],
        left_on="assignmentId",
        right_on="id",
        how="left",
    )

    submissions["studentId"] = submissions["studentId"].astype(str)

    submitted_per_user_course = (
        submissions.groupby(["studentId", "courseId"], dropna=False)["assignmentId"].nunique().reset_index()
    )
    submitted_per_user_course = submitted_per_user_course.rename(columns={"assignmentId": "submittedAssignments"})

    completion = submitted_per_user_course.merge(assignments_per_course, on="courseId", how="left")
    completion["assignmentCompletionRate"] = completion["submittedAssignments"] / completion["totalAssignments"]

    return completion, assignments


def build_attendance(live_session_assigned: pd.DataFrame, live_session_attendance: pd.DataFrame, live_sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "id" in live_sessions.columns and "liveSessionId" not in live_sessions.columns:
        live_sessions = live_sessions.rename(columns={"id": "liveSessionId"})

    assigned = live_session_assigned.groupby("userId", dropna=False)["liveSessionId"].nunique().reset_index()
    assigned = assigned.rename(columns={"liveSessionId": "assignedSessions"})

    attended = live_session_attendance.groupby("studentId", dropna=False)["liveSessionId"].nunique().reset_index()
    attended = attended.rename(columns={"studentId": "userId", "liveSessionId": "attendedSessions"})

    attendance = assigned.merge(attended, on="userId", how="left")
    attendance["attendedSessions"] = attendance["attendedSessions"].fillna(0)
    attendance["attendanceRate"] = attendance["attendedSessions"] / attendance["assignedSessions"].replace(0, np.nan)

    session_assigned = live_session_assigned.groupby("liveSessionId")["userId"].nunique().reset_index()
    session_assigned = session_assigned.rename(columns={"userId": "assignedCount"})

    session_attended = live_session_attendance.groupby("liveSessionId")["studentId"].nunique().reset_index()
    session_attended = session_attended.rename(columns={"studentId": "attendedCount"})

    session_summary = live_sessions.merge(session_assigned, on="liveSessionId", how="left")
    session_summary = session_summary.merge(session_attended, on="liveSessionId", how="left")
    session_summary["assignedCount"] = session_summary["assignedCount"].fillna(0)
    session_summary["attendedCount"] = session_summary["attendedCount"].fillna(0)
    session_summary["joinRate"] = session_summary["attendedCount"] / session_summary["assignedCount"].replace(0, np.nan)

    first_attendance = live_session_attendance.groupby("studentId")["attendedAt"].min().reset_index()
    live_session_attendance = live_session_attendance.merge(first_attendance, on="studentId", how="left", suffixes=("", "_first"))
    live_session_attendance["isNewFace"] = live_session_attendance["attendedAt"] == live_session_attendance["attendedAt_first"]

    new_faces = live_session_attendance.groupby("liveSessionId")["isNewFace"].sum().reset_index()
    new_faces = new_faces.rename(columns={"isNewFace": "newFaces"})

    session_new_faces = session_summary.merge(new_faces, on="liveSessionId", how="left")
    session_new_faces["newFaces"] = session_new_faces["newFaces"].fillna(0)
    session_new_faces["newFaceRate"] = session_new_faces["newFaces"] / session_new_faces["attendedCount"].replace(0, np.nan)

    return attendance, session_new_faces


def build_completion_and_absconded(
    enrollments: pd.DataFrame,
    assignment_completion: pd.DataFrame,
    attendance: pd.DataFrame,
    login_history: pd.DataFrame,
    max_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completion = enrollments[["userId", "courseId"]].drop_duplicates()
    completion = completion.merge(
        assignment_completion,
        left_on=["userId", "courseId"],
        right_on=["studentId", "courseId"],
        how="left",
    )
    completion = completion.merge(attendance, on="userId", how="left")

    completion["assignmentCompletionRate"] = completion["assignmentCompletionRate"].fillna(0)
    completion["attendanceRate"] = completion["attendanceRate"].fillna(0)

    completion["isComplete"] = (completion["assignmentCompletionRate"] >= 0.7) & (completion["attendanceRate"] >= 0.7)

    login_history = to_datetime(login_history, ["timestamp"])
    login_success = login_history[login_history["status"] == "success"]
    last_login = login_success.groupby("userId")["timestamp"].max().reset_index()

    absconded = completion.merge(last_login, on="userId", how="left")
    cutoff = max_date - pd.Timedelta(days=30)
    absconded["inactive30"] = absconded["timestamp"].isna() | (absconded["timestamp"] < cutoff)
    absconded["isAbsconded"] = (
        (absconded["assignmentCompletionRate"] < 0.1)
        & (absconded["attendanceRate"] < 0.1)
        & absconded["inactive30"]
    )

    return completion, absconded


def instructor_performance(live_sessions: pd.DataFrame, live_session_assigned: pd.DataFrame, live_session_attendance: pd.DataFrame) -> pd.DataFrame:
    if "id" in live_sessions.columns and "liveSessionId" not in live_sessions.columns:
        live_sessions = live_sessions.rename(columns={"id": "liveSessionId"})

    assigned = live_session_assigned.groupby("liveSessionId")["userId"].nunique().reset_index().rename(columns={"userId": "assignedCount"})
    attended = live_session_attendance.groupby("liveSessionId")["studentId"].nunique().reset_index().rename(columns={"studentId": "attendedCount"})

    merged = live_sessions.merge(assigned, on="liveSessionId", how="left").merge(attended, on="liveSessionId", how="left")
    merged["assignedCount"] = merged["assignedCount"].fillna(0)
    merged["attendedCount"] = merged["attendedCount"].fillna(0)
    merged["joinRate"] = merged["attendedCount"] / merged["assignedCount"].replace(0, np.nan)

    by_instructor = merged.groupby("createdById").agg(
        sessions=("liveSessionId", "nunique"),
        assigned=("assignedCount", "sum"),
        attended=("attendedCount", "sum"),
        join_rate=("joinRate", "mean"),
    ).reset_index()

    return by_instructor


def buyers_remorse_window(
    payments: pd.DataFrame,
    live_session_attendance: pd.DataFrame,
    assignment_submissions: pd.DataFrame,
    login_history: pd.DataFrame,
) -> pd.DataFrame:
    payments = payments[payments["status"] == "succeeded"].copy()
    payments = payments.dropna(subset=["paidAt"])

    attendance = live_session_attendance.copy()
    attendance = attendance.rename(columns={"studentId": "userId"})

    assignment_submissions = assignment_submissions.copy()
    assignment_submissions = assignment_submissions.rename(columns={"studentId": "userId"})

    login_success = login_history[login_history["status"] == "success"].copy()

    rows = []
    for _, pay in payments.iterrows():
        user_id = pay.get("userId")
        paid_at = pay.get("paidAt")
        if pd.isna(paid_at):
            continue

        def _count_between(df: pd.DataFrame, time_col: str, start: int, end: int) -> int:
            mask = (df["userId"] == user_id) & (df[time_col] >= paid_at + pd.Timedelta(days=start)) & (df[time_col] <= paid_at + pd.Timedelta(days=end))
            return int(mask.sum())

        week1_attendance = _count_between(attendance, "attendedAt", 0, 7)
        week4_attendance = _count_between(attendance, "attendedAt", 22, 28)
        week1_submissions = _count_between(assignment_submissions, "submittedAt", 0, 7)
        week4_submissions = _count_between(assignment_submissions, "submittedAt", 22, 28)
        week1_logins = _count_between(login_success, "timestamp", 0, 7)
        week4_logins = _count_between(login_success, "timestamp", 22, 28)

        rows.append(
            {
                "userId": user_id,
                "paidAt": paid_at,
                "week1_attendance": week1_attendance,
                "week4_attendance": week4_attendance,
                "week1_submissions": week1_submissions,
                "week4_submissions": week4_submissions,
                "week1_logins": week1_logins,
                "week4_logins": week4_logins,
            }
        )

    return pd.DataFrame(rows)


def agreement_compliance_time(assignments: pd.DataFrame, agreements: pd.DataFrame) -> pd.DataFrame:
    assignments = assignments.copy()
    assignments["publishedAt"] = assignments["publishedAt"].fillna(assignments["createdAt"])

    agreements = agreements.copy()
    agreements = agreements.rename(columns={"studentId": "userId"})

    merged = agreements.merge(assignments[["id", "publishedAt"]], left_on="assignmentId", right_on="id", how="left")
    merged = merged.dropna(subset=["agreedAt", "publishedAt"])
    merged["complianceHours"] = (merged["agreedAt"] - merged["publishedAt"]).dt.total_seconds() / 3600

    summary = merged.groupby("assignmentId")["complianceHours"].agg(["count", "mean", "median"]).reset_index()
    return summary
