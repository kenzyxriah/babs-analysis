from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.io.loaders import month_start
from analytics.io.writers import ensure_dirs, save_table
from analytics.models.schema import Context
from analytics.features.lead_nlp import parse_form_submissions, extract_skill_gap_llm
from analytics.features.engagement import (
    build_assignment_completion,
    build_attendance,
    build_completion_and_absconded,
    instructor_performance,
    buyers_remorse_window,
    agreement_compliance_time,
)
from analytics.features.finance import (
    payment_status_by_month,
    revenue_by_month,
    payment_delinquency,
    paid_in_full_by_product,
    payment_plan_default_rate,
    commitment_vs_cash,
    discount_hook_summary,
    payment_plan_engagement,
    investment_vs_engagement,
)
from analytics.features.products import (
    course_product_map,
    gateway_upgrade,
    product_revenue_pareto,
    module_saturation,
)
from analytics.features.ops import (
    ops_gap_report,
    exception_duration_summary,
    exception_timeline,
    sales_lag,
    golden_layer_correlations,
)
from analytics.features.gateway_attribution import classify_gateway_sessions


async def build_tables(ctx: Context) -> None:
    settings = ctx.settings
    data = ctx.data

    ensure_dirs(settings.table_dir, settings.fig_dir)

    users = data["users"]
    roles = data["roles"]
    courses = data["courses"]
    modules = data["modules"]
    assignments = data["assignments"]
    assignment_submissions = data["assignment_submissions"]
    assignment_agreements = data["assignment_user_agreements"]
    module_assigned_users = data["module_assigned_users"]
    live_sessions = data["live_sessions"]
    live_session_assigned = data["live_session_assigned_students"]
    live_session_attendance = data["live_session_attendance"]
    products = data["products"]
    product_assets = data["product_assets"]
    product_accesses = data["product_accesses"]
    payments = data["payments"]
    payment_commitments = data["payment_commitments"]
    payment_agreements = data["payment_agreements"]
    payment_exceptions = data["payment_exceptions"]
    programs = data["programs"]
    program_courses = data["program_courses"]
    product_programs = data["product_programs"]
    user_program_selections = data["user_program_selections"]
    tags = data["tags"]
    categories = data["categories"]
    course_tags = data["course_tags"]
    product_tags = data["product_tags"]
    program_tags = data["program_tags"]
    custom_products = data["custom_products"]
    forms = data["form"]
    form_submissions = data["form_submission"]
    login_history = data["login_history"]

    student_role_ids = set()
    if not roles.empty and "name" in roles.columns:
        student_role_ids = set(roles[roles["name"].str.lower() == "student"]["id"].tolist())
    if not student_role_ids:
        student_role_ids = {2}
    student_ids = set(users[users["roleId"].isin(student_role_ids)]["id"].tolist()) if not users.empty else set()

    # Filter payment-related datasets to student users only (exclude admin/test activity).
    if student_ids:
        payments = payments[payments["userId"].isin(student_ids)].copy()
        payment_commitments = payment_commitments[payment_commitments["userId"].isin(student_ids)].copy()
        payment_agreements = payment_agreements[payment_agreements["userId"].isin(student_ids)].copy()
        payment_exceptions = payment_exceptions[payment_exceptions["userId"].isin(student_ids)].copy()
        custom_products = custom_products[custom_products["userId"].isin(student_ids)].copy()

    course_titles = courses[["id", "title"]].rename(columns={"id": "courseId", "title": "courseTitle"})
    product_titles = products[["id", "title", "price", "discountPrice"]].rename(columns={"id": "productId", "title": "productTitle"})

    course_products = course_product_map(product_assets)
    enrollments = product_accesses.merge(course_products, on="productId", how="left")
    enrollments["enrollmentDate"] = enrollments["startDate"].fillna(enrollments["createdAt"])

    # Adoption
    enrollments["enrollmentMonth"] = month_start(enrollments["enrollmentDate"])
    enrollments_by_course_month = enrollments.groupby(["courseId", "enrollmentMonth"], dropna=False)["userId"].nunique().reset_index()
    enrollments_by_course_month = enrollments_by_course_month.merge(course_titles, on="courseId", how="left")
    if "courseTitle" in enrollments_by_course_month.columns:
        enrollments_by_course_month = enrollments_by_course_month[["courseTitle", "courseId", "enrollmentMonth", "userId"]]
    save_table(enrollments_by_course_month, settings.table_dir / "enrollments_by_course_month.csv")
    ctx.add_result("enrollments_by_course_month", enrollments_by_course_month)

    enrollments_by_product_month = enrollments.groupby(["productId", "enrollmentMonth"], dropna=False)["userId"].nunique().reset_index()
    enrollments_by_product_month = enrollments_by_product_month.merge(product_titles, on="productId", how="left")
    if "productTitle" in enrollments_by_product_month.columns:
        enrollments_by_product_month = enrollments_by_product_month[["productTitle", "productId", "enrollmentMonth", "userId", "price", "discountPrice"]]
    save_table(enrollments_by_product_month, settings.table_dir / "enrollments_by_product_month.csv")
    ctx.add_result("enrollments_by_product_month", enrollments_by_product_month)

    total_users = users["id"].nunique() if not users.empty else 0
    adoption_summary = product_accesses.groupby("productId").agg(
        unique_users=("userId", "nunique"),
        active_users=("isActive", lambda s: (s == 1).sum()),
    ).reset_index()
    adoption_summary["adoption_rate"] = adoption_summary["unique_users"] / total_users if total_users else np.nan
    adoption_summary = adoption_summary.merge(product_titles, on="productId", how="left")
    save_table(adoption_summary, settings.table_dir / "product_adoption_summary.csv")
    ctx.add_result("product_adoption_summary", adoption_summary)

    # Program selection and catalog taxonomy
    program_titles = programs[["id", "title"]].rename(columns={"id": "programId", "title": "programTitle"})
    program_selection = user_program_selections.copy()
    if not program_selection.empty:
        level_counts = program_selection.groupby(["programId", "level"])["userId"].nunique().unstack(fill_value=0)
        level_counts["major_users"] = level_counts.get("major", 0)
        level_counts["minor_users"] = level_counts.get("minor", 0)
        level_counts["selected_users"] = level_counts["major_users"] + level_counts["minor_users"]
        level_counts = level_counts.reset_index()
    else:
        level_counts = pd.DataFrame(columns=["programId", "major_users", "minor_users", "selected_users"])

    courses_per_program = program_courses.groupby("programId")["courseId"].nunique().rename("linked_courses").reset_index()
    products_per_program = product_programs.groupby("programId")["productId"].nunique().rename("linked_products").reset_index()

    program_summary = level_counts.merge(courses_per_program, on="programId", how="left")
    program_summary = program_summary.merge(products_per_program, on="programId", how="left")
    program_summary = program_summary.merge(program_titles, on="programId", how="left")
    program_summary["linked_courses"] = program_summary["linked_courses"].fillna(0)
    program_summary["linked_products"] = program_summary["linked_products"].fillna(0)
    program_summary["major_share"] = program_summary["major_users"] / program_summary["selected_users"].replace(0, np.nan)
    program_summary["selection_share"] = program_summary["selected_users"] / total_users if total_users else np.nan
    save_table(program_summary, settings.table_dir / "program_selection_summary.csv")
    ctx.add_result("program_selection_summary", program_summary)

    # Tag category coverage by entity type
    tag_categories = tags.merge(categories[["id", "name"]], left_on="categoryId", right_on="id", how="left", suffixes=("", "_category"))
    def _tag_coverage(entity_tags: pd.DataFrame, entity_col: str, entity_type: str) -> pd.DataFrame:
        if entity_tags.empty:
            return pd.DataFrame(columns=["category", "entities", "entityType"])
        merged = entity_tags.merge(tag_categories[["id", "name"]], left_on="tagId", right_on="id", how="left")
        coverage = merged.groupby("name")[entity_col].nunique().reset_index().rename(columns={"name": "category", entity_col: "entities"})
        coverage["entityType"] = entity_type
        return coverage

    tag_coverage = pd.concat(
        [
            _tag_coverage(product_tags, "productId", "Products"),
            _tag_coverage(course_tags, "courseId", "Courses"),
            _tag_coverage(program_tags, "programId", "Programs"),
        ],
        ignore_index=True,
    )
    save_table(tag_coverage, settings.table_dir / "tag_category_coverage.csv")
    ctx.add_result("tag_category_coverage", tag_coverage)

    # Specialization tag revenue (attributed across tags per product)
    spec_ids = categories[categories["name"].str.lower() == "specialization"]["id"].tolist()
    if spec_ids:
        spec_tags = tags[tags["categoryId"].isin(spec_ids)][["id", "name"]]
    else:
        spec_tags = tags.iloc[0:0][["id", "name"]]

    if not spec_tags.empty:
        spec_map = product_tags.merge(spec_tags, left_on="tagId", right_on="id", how="left").dropna(subset=["name"])
        tag_counts = spec_map.groupby("productId")["tagId"].nunique().rename("tag_count").reset_index()
        paid = payments[(payments["paidAt"].notna()) | (payments["status"].str.lower() == "succeeded")].copy()
        paid_revenue = paid.groupby("productId")["amount"].sum().reset_index().rename(columns={"amount": "paidRevenue"})
        paid_revenue = paid_revenue.merge(tag_counts, on="productId", how="left")
        paid_revenue["tag_count"] = paid_revenue["tag_count"].replace(0, np.nan)
        paid_revenue["rev_per_tag"] = paid_revenue["paidRevenue"] / paid_revenue["tag_count"]
        spec_map = spec_map.merge(paid_revenue[["productId", "rev_per_tag"]], on="productId", how="left")
        spec_rev = spec_map.groupby("name").agg(
            attributed_paid_revenue=("rev_per_tag", "sum"),
            tagged_products=("productId", "nunique"),
        ).reset_index().rename(columns={"name": "tag"})
    else:
        spec_rev = pd.DataFrame(columns=["tag", "attributed_paid_revenue", "tagged_products"])
    save_table(spec_rev, settings.table_dir / "specialization_tag_revenue.csv")
    ctx.add_result("specialization_tag_revenue", spec_rev)

    # Payment health
    status_by_month = payment_status_by_month(payments)
    save_table(status_by_month, settings.table_dir / "payments_status_by_month.csv")
    ctx.add_result("payments_status_by_month", status_by_month)

    revenue = revenue_by_month(payments)
    save_table(revenue, settings.table_dir / "revenue_by_month.csv")
    ctx.add_result("revenue_by_month", revenue)

    custom_rev = custom_products.dropna(subset=["createdAt"]).copy()
    if not custom_rev.empty:
        custom_rev["revenueMonth"] = month_start(custom_rev["createdAt"])
        custom_rev = custom_rev.groupby("revenueMonth")["totalPrice"].sum().reset_index().rename(columns={"totalPrice": "revenue"})
    else:
        custom_rev = pd.DataFrame(columns=["revenueMonth", "revenue"])
    save_table(custom_rev, settings.table_dir / "custom_product_revenue_by_month.csv")
    ctx.add_result("custom_product_revenue_by_month", custom_rev)

    paid = payments[(payments["paidAt"].notna()) | (payments["status"].str.lower() == "succeeded")].copy()
    if not paid.empty:
        paid["paidMonth"] = month_start(paid["paidAt"].fillna(paid["createdAt"]))
        payments_received = paid.groupby("paidMonth")["amount"].sum().reset_index().rename(columns={"amount": "payments"})
    else:
        payments_received = pd.DataFrame(columns=["paidMonth", "payments"])
    save_table(payments_received, settings.table_dir / "payments_received_by_month.csv")
    ctx.add_result("payments_received_by_month", payments_received)

    paid_revenue = paid.groupby("productId")["amount"].sum().reset_index().rename(columns={"amount": "paidRevenue"})
    paid_revenue = paid_revenue.merge(product_titles, on="productId", how="left")
    save_table(paid_revenue, settings.table_dir / "paid_revenue_by_product.csv")
    ctx.add_result("paid_revenue_by_product", paid_revenue)

    max_date = max(
        payments["createdAt"].max(),
        login_history["timestamp"].max(),
        product_accesses["createdAt"].max(),
        live_session_attendance["attendedAt"].max() if "attendedAt" in live_session_attendance else pd.Timestamp.min,
    )
    delinquency = payment_delinquency(payments, max_date)
    save_table(delinquency, settings.table_dir / "payment_delinquency.csv")
    ctx.add_result("payment_delinquency", delinquency)

    paid_in_full = paid_in_full_by_product(payments).merge(product_titles, on="productId", how="left")
    save_table(paid_in_full, settings.table_dir / "paid_in_full_by_product.csv")
    ctx.add_result("paid_in_full_by_product", paid_in_full)

    # Completion and attendance
    assignment_completion, assignments_with_course = build_assignment_completion(assignments, assignment_submissions, modules)
    assignment_completion_by_course = assignment_completion.groupby("courseId")["assignmentCompletionRate"].mean().reset_index().merge(course_titles, on="courseId", how="left")
    save_table(assignment_completion_by_course, settings.table_dir / "assignment_completion_by_course.csv")
    ctx.add_result("assignment_completion_by_course", assignment_completion_by_course)

    attendance, session_new_faces = build_attendance(live_session_assigned, live_session_attendance, live_sessions)
    session_titles = live_sessions.rename(columns={"id": "liveSessionId", "title": "sessionTitle"})[["liveSessionId", "sessionTitle", "scheduledAt", "createdById"]]
    session_new_faces = session_new_faces.merge(session_titles, on="liveSessionId", how="left")
    # Normalize scheduledAt column
    if "scheduledAt" not in session_new_faces.columns:
        if "scheduledAt_y" in session_new_faces.columns:
            session_new_faces["scheduledAt"] = session_new_faces["scheduledAt_y"]
        elif "scheduledAt_x" in session_new_faces.columns:
            session_new_faces["scheduledAt"] = session_new_faces["scheduledAt_x"]
    # Clean up columns for exec-friendly table
    session_summary_clean = session_new_faces[
        [
            "liveSessionId",
            "sessionTitle",
            "scheduledAt",
            "assignedCount",
            "attendedCount",
            "joinRate",
            "newFaces",
            "newFaceRate",
        ]
    ]
    save_table(session_summary_clean, settings.table_dir / "session_attendance_summary.csv")
    ctx.add_result("session_attendance_summary", session_summary_clean)

    # Engagement trends over time (attendance + submissions)
    att = live_session_attendance.copy()
    att["attendMonth"] = month_start(att["attendedAt"])
    att_month = att.groupby("attendMonth")["id"].nunique().reset_index().rename(columns={"id": "attendanceEvents"})

    subs = assignment_submissions.copy()
    subs["submitMonth"] = month_start(subs["submittedAt"])
    subs_month = subs.groupby("submitMonth")["id"].nunique().reset_index().rename(columns={"id": "submissionEvents"})

    engagement_trends = att_month.merge(subs_month, left_on="attendMonth", right_on="submitMonth", how="outer")
    engagement_trends["month"] = engagement_trends["attendMonth"].fillna(engagement_trends["submitMonth"])
    engagement_trends = engagement_trends.drop(columns=["attendMonth", "submitMonth"])
    engagement_trends = engagement_trends.sort_values("month")
    save_table(engagement_trends, settings.table_dir / "engagement_trends_over_time.csv")
    ctx.add_result("engagement_trends_over_time", engagement_trends)

    # Join rate trends (gateway vs non-gateway sessions)
    session_flags = classify_gateway_sessions(live_sessions)
    session_att = live_session_assigned.groupby("liveSessionId")["userId"].nunique().reset_index().rename(columns={"userId": "assignedCount"})
    session_attend = live_session_attendance.groupby("liveSessionId")["studentId"].nunique().reset_index().rename(columns={"studentId": "attendedCount"})
    session_join = session_flags.merge(session_att, on="liveSessionId", how="left").merge(session_attend, on="liveSessionId", how="left")
    session_join["assignedCount"] = session_join["assignedCount"].fillna(0)
    session_join["attendedCount"] = session_join["attendedCount"].fillna(0)
    session_join["joinRate"] = session_join["attendedCount"] / session_join["assignedCount"].replace(0, np.nan)
    session_join["sessionMonth"] = month_start(session_join["scheduledAt"])
    join_trends = session_join.groupby(["sessionMonth", "is_gateway_session"]).agg(
        avg_join_rate=("joinRate", "mean"),
        sessions=("liveSessionId", "nunique"),
    ).reset_index()
    join_trends["sessionType"] = np.where(join_trends["is_gateway_session"], "Gateway", "Non-Gateway")
    save_table(join_trends, settings.table_dir / "session_join_rate_trends.csv")
    ctx.add_result("session_join_rate_trends", join_trends)

    completion, absconded = build_completion_and_absconded(enrollments, assignment_completion, attendance, login_history, max_date)
    # Keep detailed enrollment-level completion in-memory for weighted headline metrics (do not write to disk).
    ctx.add_result("completion_detail", completion)
    ctx.add_result("absconded_detail", absconded)

    # Context table: why "strict completion" can be 0% (separate thresholds vs combined).
    completion_flags = completion.copy()
    completion_flags["meet_attendance_70"] = completion_flags["attendanceRate"] >= 0.7
    completion_flags["meet_assignments_70"] = completion_flags["assignmentCompletionRate"] >= 0.7
    completion_flags["meet_both_70"] = completion_flags["meet_attendance_70"] & completion_flags["meet_assignments_70"]
    threshold_breakdown = pd.DataFrame(
        {
            "metric": ["Meet attendance >=70%", "Meet assignments >=70%", "Meet both (strict completion)"],
            "users": [
                int(completion_flags["meet_attendance_70"].sum()),
                int(completion_flags["meet_assignments_70"].sum()),
                int(completion_flags["meet_both_70"].sum()),
            ],
            "rate": [
                float(completion_flags["meet_attendance_70"].mean()) if not completion_flags.empty else np.nan,
                float(completion_flags["meet_assignments_70"].mean()) if not completion_flags.empty else np.nan,
                float(completion_flags["meet_both_70"].mean()) if not completion_flags.empty else np.nan,
            ],
        }
    )
    save_table(threshold_breakdown, settings.table_dir / "completion_threshold_breakdown.csv")
    ctx.add_result("completion_threshold_breakdown", threshold_breakdown)
    completion_summary = completion.groupby("courseId")["isComplete"].mean().reset_index().rename(columns={"isComplete": "completionRate"})
    completion_summary = completion_summary.merge(course_titles, on="courseId", how="left")
    if "courseTitle" in completion_summary.columns:
        completion_summary = completion_summary[["courseTitle", "courseId", "completionRate"]]
    save_table(completion_summary, settings.table_dir / "course_completion_summary.csv")
    ctx.add_result("course_completion_summary", completion_summary)

    absconded_by_course = absconded.groupby("courseId")["isAbsconded"].mean().reset_index().rename(columns={"isAbsconded": "abscondRate"})
    absconded_by_course = absconded_by_course.merge(course_titles, on="courseId", how="left")
    if "courseTitle" in absconded_by_course.columns:
        absconded_by_course = absconded_by_course[["courseTitle", "courseId", "abscondRate"]]
    save_table(absconded_by_course, settings.table_dir / "absconded_by_course.csv")
    ctx.add_result("absconded_by_course", absconded_by_course)

    # Assignment submissions summary
    assigned_users_by_module = module_assigned_users.groupby("moduleId")["userId"].nunique().reset_index().rename(columns={"userId": "assignedUsers"})
    submission_counts = assignment_submissions.groupby("assignmentId")["studentId"].nunique().reset_index().rename(columns={"studentId": "submissions"})
    assignment_submission_summary = assignments.merge(submission_counts, left_on="id", right_on="assignmentId", how="left")
    assignment_submission_summary = assignment_submission_summary.merge(assigned_users_by_module, on="moduleId", how="left")
    assignment_submission_summary["submissions"] = assignment_submission_summary["submissions"].fillna(0)
    assignment_submission_summary["assignedUsers"] = assignment_submission_summary["assignedUsers"].fillna(0)
    assignment_submission_summary["submissionRate"] = assignment_submission_summary["submissions"] / assignment_submission_summary["assignedUsers"].replace(0, np.nan)
    save_table(assignment_submission_summary[["id", "title", "moduleId", "submissions", "assignedUsers", "submissionRate"]], settings.table_dir / "assignment_submission_summary.csv")
    ctx.add_result("assignment_submission_summary", assignment_submission_summary)

    time_to_submit = assignment_submissions.merge(
        assignments[["id", "publishedAt", "dueDate"]],
        left_on="assignmentId",
        right_on="id",
        how="left",
    )
    time_to_submit["baselineAt"] = time_to_submit["dueDate"].where(time_to_submit["dueDate"].notna(), time_to_submit["publishedAt"])
    time_to_submit["time_to_submit_hours"] = (time_to_submit["submittedAt"] - time_to_submit["baselineAt"]).dt.total_seconds() / 3600
    time_to_submit = time_to_submit[["assignmentId", "time_to_submit_hours"]].dropna(subset=["time_to_submit_hours"])
    save_table(time_to_submit, settings.table_dir / "time_to_submit_distribution.csv")
    ctx.add_result("time_to_submit_distribution", time_to_submit)

    grading_latency = assignment_submissions.dropna(subset=["submittedAt", "gradedAt"]).copy()
    if not grading_latency.empty:
        grading_latency["gradingHours"] = (grading_latency["gradedAt"] - grading_latency["submittedAt"]).dt.total_seconds() / 3600
        grading_latency_summary = grading_latency.groupby("assignmentId")["gradingHours"].agg(["count", "mean", "median"]).reset_index()
        grading_latency_summary = grading_latency_summary.merge(assignments[["id", "title"]], left_on="assignmentId", right_on="id", how="left")
        save_table(grading_latency_summary[["assignmentId", "title", "count", "mean", "median"]], settings.table_dir / "grading_latency.csv")
        ctx.add_result("grading_latency", grading_latency_summary)

    # Login engagement
    login_success = login_history[login_history["status"] == "success"].copy()
    login_success["loginDate"] = login_success["timestamp"].dt.date
    daily_active = login_success.groupby("loginDate")["userId"].nunique().reset_index().rename(columns={"userId": "DAU"})
    save_table(daily_active, settings.table_dir / "login_daily_active.csv")
    ctx.add_result("login_daily_active", daily_active)

    login_success["loginMonth"] = month_start(login_success["timestamp"])
    monthly_active = login_success.groupby("loginMonth")["userId"].nunique().reset_index().rename(columns={"userId": "MAU"})
    save_table(monthly_active, settings.table_dir / "login_monthly_active.csv")
    ctx.add_result("login_monthly_active", monthly_active)

    # Leads
    leads = parse_form_submissions(form_submissions, forms)
    leads_by_form = leads.groupby("formTitle")["submissionId"].nunique().reset_index().rename(columns={"submissionId": "leadCount"})
    save_table(leads_by_form, settings.table_dir / "inquiry_volume_by_form.csv")
    ctx.add_result("inquiry_volume_by_form", leads_by_form)

    leads["leadMonth"] = month_start(leads["submittedAt"])
    leads_by_month = leads.groupby("leadMonth")["submissionId"].nunique().reset_index().rename(columns={"submissionId": "leadCount"})
    save_table(leads_by_month, settings.table_dir / "inquiry_volume_by_month.csv")
    ctx.add_result("inquiry_volume_by_month", leads_by_month)

    lead_intent = leads.groupby("intentCategory")["submissionId"].nunique().reset_index().rename(columns={"submissionId": "leadCount"})
    save_table(lead_intent, settings.table_dir / "inquiry_intent.csv")
    ctx.add_result("inquiry_intent", lead_intent)

    lead_tags = leads.copy()
    lead_tags["intentTagsList"] = lead_tags["intentTags"].fillna("").str.split(";")
    lead_tags = lead_tags.explode("intentTagsList")
    lead_tags = lead_tags[lead_tags["intentTagsList"].notna() & (lead_tags["intentTagsList"] != "")]
    lead_tag_counts = lead_tags.groupby("intentTagsList")["submissionId"].nunique().reset_index().rename(columns={"intentTagsList": "intentTag", "submissionId": "leadCount"})
    lead_tag_counts["share"] = lead_tag_counts["leadCount"] / lead_tag_counts["leadCount"].sum()
    save_table(lead_tag_counts, settings.table_dir / "inquiry_intent_tags.csv")
    ctx.add_result("inquiry_intent_tags", lead_tag_counts)

    def bucket_goal(text: object) -> str:
        if not isinstance(text, str) or not text.strip():
            return "none"
        t = text.lower()
        buckets = {
            "data science/analytics": ["data science", "data scientist", "data analyst", "analytics"],
            "salesforce": ["salesforce", "admin", "administrator", "crm"],
            "business analysis": ["business analyst", "business analysis", "ba"],
            "product management": ["product manager", "product management", "product owner"],
            "cloud/devops": ["cloud", "devops", "aws", "azure", "gcp"],
            "cybersecurity": ["cyber", "security", "infosec"],
            "scrum/agile": ["scrum", "agile", "scrum master"],
            "software engineering": ["software", "developer", "engineer", "programmer"],
        }
        for bucket, keys in buckets.items():
            if any(k in t for k in keys):
                return bucket
        return "other"

    goal_buckets = leads.copy()
    goal_buckets["goalBucket"] = goal_buckets["careerGoal"].apply(bucket_goal)
    goal_counts = goal_buckets.groupby("goalBucket")["submissionId"].nunique().reset_index().rename(columns={"submissionId": "count"})
    goal_counts["share"] = goal_counts["count"] / goal_counts["count"].sum()
    save_table(goal_counts, settings.table_dir / "career_goal_buckets.csv")
    ctx.add_result("career_goal_buckets", goal_counts)

    leads = leads.merge(users[["id", "email"]], left_on="email", right_on="email", how="left")
    leads["isUser"] = leads["id"].notna()
    paid_users = payments[payments["status"] == "succeeded"]["userId"].unique().tolist()
    leads["isPaidUser"] = leads["id"].isin(paid_users)
    lead_conversion = leads.groupby("formTitle").agg(
        leads=("submissionId", "nunique"),
        users=("isUser", "sum"),
        paid_users=("isPaidUser", "sum"),
    ).reset_index()
    save_table(lead_conversion, settings.table_dir / "inquiry_conversion.csv")
    ctx.add_result("inquiry_conversion", lead_conversion)

    # LLM skill gap extraction
    leads_llm_input = leads.rename(columns={"id": "userId"})
    skill_gap = await extract_skill_gap_llm(
        leads_llm_input,
        settings.groq_api_key,
        settings.groq_model,
        settings.max_llm_rows,
        settings.llm_batch_size,
        settings.llm_batch_sleep_seconds,
    )
    save_table(skill_gap, settings.table_dir / "skill_gap_extractions.csv")
    ctx.add_result("skill_gap_extractions", skill_gap)

    # Instructor performance
    instructor = instructor_performance(live_sessions, live_session_assigned, live_session_attendance)
    # Add human-readable instructor names
    instructor_name = users[["id", "firstName", "lastName"]].copy()
    instructor_name["instructorName"] = (
        instructor_name["firstName"].fillna("").astype(str).str.strip()
        + " "
        + instructor_name["lastName"].fillna("").astype(str).str.strip()
    ).str.strip()
    instructor = instructor.merge(instructor_name[["id", "instructorName"]], left_on="createdById", right_on="id", how="left")
    instructor = instructor.drop(columns=["id"])
    save_table(instructor, settings.table_dir / "instructor_performance.csv")
    ctx.add_result("instructor_performance", instructor)

    # Buyer remorse
    remorse = buyers_remorse_window(payments, live_session_attendance, assignment_submissions, login_history)
    save_table(remorse, settings.table_dir / "buyers_remorse_window.csv")
    ctx.add_result("buyers_remorse_window", remorse)

    # Career goal vs spend
    spend = payments[payments["status"] == "succeeded"].groupby("userId")["amount"].sum().reset_index()
    career_spend = leads_llm_input.merge(spend, on="userId", how="left")
    career_goal_spend = career_spend.groupby("careerGoal")["amount"].mean().reset_index().rename(columns={"amount": "avgSpend"})
    save_table(career_goal_spend, settings.table_dir / "career_goal_spend.csv")
    ctx.add_result("career_goal_spend", career_goal_spend)

    # Payment plan engagement
    payment_plan = payment_plan_engagement(assignment_submissions, payments, payment_commitments, custom_products)
    save_table(payment_plan, settings.table_dir / "payment_plan_engagement.csv")
    ctx.add_result("payment_plan_engagement", payment_plan)

    # Sales lag
    sales_lag_df = sales_lag(leads, users, payments)
    save_table(sales_lag_df, settings.table_dir / "sales_lag_distribution.csv")
    ctx.add_result("sales_lag_distribution", sales_lag_df)

    # Agreement compliance
    agreement_dist = assignment_agreements.merge(
        assignments[["id", "publishedAt", "createdAt"]],
        left_on="assignmentId",
        right_on="id",
        how="left",
    )
    agreement_dist["publishedAt"] = agreement_dist["publishedAt"].fillna(agreement_dist["createdAt"])
    agreement_dist = agreement_dist.dropna(subset=["agreedAt", "publishedAt"])
    agreement_dist["hoursToAgree"] = (agreement_dist["agreedAt"] - agreement_dist["publishedAt"]).dt.total_seconds() / 3600
    agreement_dist = agreement_dist[["assignmentId", "hoursToAgree"]]
    save_table(agreement_dist, settings.table_dir / "agreement_compliance_distribution.csv")
    ctx.add_result("agreement_compliance_distribution", agreement_dist)

    agreement_time = agreement_compliance_time(assignments, assignment_agreements)
    agreement_time = agreement_time.merge(assignments[["id", "title"]], left_on="assignmentId", right_on="id", how="left").drop(columns=["id"])
    save_table(agreement_time, settings.table_dir / "agreement_compliance_time.csv")
    ctx.add_result("agreement_compliance_time", agreement_time)

    # Commitment vs cash
    waterfall = commitment_vs_cash(payments, payment_commitments, custom_products)
    save_table(waterfall, settings.table_dir / "revenue_waterfall.csv")
    ctx.add_result("revenue_waterfall", waterfall)

    # Payment plan default rate
    default_rate = payment_plan_default_rate(payment_agreements, payment_commitments)
    save_table(default_rate, settings.table_dir / "payment_plan_default_rate.csv")
    ctx.add_result("payment_plan_default_rate", default_rate)

    # Exceptions
    exception_summary = exception_duration_summary(payment_exceptions)
    save_table(exception_summary, settings.table_dir / "exception_duration_summary.csv")
    ctx.add_result("exception_duration_summary", exception_summary)

    exception_timeline_df = exception_timeline(payment_exceptions)
    save_table(exception_timeline_df, settings.table_dir / "exception_timeline.csv")
    ctx.add_result("exception_timeline", exception_timeline_df)

    # Discount hook
    discount_hook = discount_hook_summary(products, payments)
    discount_hook = discount_hook.merge(product_titles[["productId", "productTitle"]], on="productId", how="left")
    if "productTitle" in discount_hook.columns:
        discount_hook["discount_share"] = discount_hook["discount_sales"] / discount_hook["total_sales"].replace(0, np.nan)
        discount_hook = discount_hook[["productTitle", "productId", "discount_sales", "full_sales", "total_sales", "discount_share"]]
    save_table(discount_hook, settings.table_dir / "discount_hook_summary.csv")
    ctx.add_result("discount_hook_summary", discount_hook)

    # Investment vs engagement
    invest_engage = investment_vs_engagement(assignment_submissions, payments)
    save_table(invest_engage, settings.table_dir / "investment_vs_engagement.csv")
    ctx.add_result("investment_vs_engagement", invest_engage)

    # Best sellers + Pareto
    pareto = product_revenue_pareto(payments, products)
    if "productTitle" in pareto.columns:
        pareto = pareto[["productTitle", "productId", "units", "revenue", "cumulative_revenue", "cumulative_share"]]
    save_table(pareto, settings.table_dir / "product_revenue_pareto.csv")
    ctx.add_result("product_revenue_pareto", pareto)

    # Module saturation
    saturation = module_saturation(modules, module_assigned_users)
    save_table(saturation, settings.table_dir / "module_saturation.csv")
    ctx.add_result("module_saturation", saturation)

    # Gateway upgrade
    gateway_summary, gateway_timeline = gateway_upgrade(payments, products, settings.gateway_price_quantile, settings.mentorship_price_quantile)
    save_table(gateway_summary, settings.table_dir / "gateway_upgrade_summary.csv")
    save_table(gateway_timeline, settings.table_dir / "gateway_upgrade_timeline.csv")
    ctx.add_result("gateway_upgrade_summary", gateway_summary)
    ctx.add_result("gateway_upgrade_timeline", gateway_timeline)

    # Ops gaps
    ops_gaps = ops_gap_report(enrollments, payments, login_history, max_date)
    save_table(ops_gaps, settings.table_dir / "ops_gap_report.csv")
    ctx.add_result("ops_gap_report", ops_gaps)

    # Golden layer correlations
    engagement_df = assignment_completion.groupby("studentId")["assignmentCompletionRate"].mean().reset_index().rename(columns={"studentId": "userId"})
    golden = golden_layer_correlations(leads_llm_input.rename(columns={"id": "userId"}), engagement_df, payments)
    if not golden.empty:
        save_table(golden, settings.table_dir / "golden_layer_correlations.csv")
        ctx.add_result("golden_layer_correlations", golden)

    # Segment KPIs
    paid_users = payments[payments["status"] == "succeeded"]["userId"].unique()
    segment = enrollments[["userId", "productId"]].drop_duplicates().merge(products[["id", "accessType"]], left_on="productId", right_on="id", how="left")
    segment["isPaid"] = segment["userId"].isin(paid_users)
    completion_flag = completion.groupby("userId")["isComplete"].max().reset_index()
    segment = segment.merge(completion_flag, on="userId", how="left")
    segment["isComplete"] = segment["isComplete"].fillna(False)
    segment_kpis = segment.groupby("accessType").agg(
        enrollments=("userId", "nunique"),
        paid_rate=("isPaid", "mean"),
        completion_rate=("isComplete", "mean"),
    ).reset_index()
    save_table(segment_kpis, settings.table_dir / "segment_kpis.csv")
    ctx.add_result("segment_kpis", segment_kpis)

    # Users by role
    users_by_role = users.groupby("roleId")["id"].nunique().reset_index().merge(roles[["id", "name"]], left_on="roleId", right_on="id", how="left")
    users_by_role = users_by_role.rename(columns={"id_x": "userCount", "name": "role"})
    save_table(users_by_role[["roleId", "role", "userCount"]], settings.table_dir / "users_by_role.csv")
    ctx.add_result("users_by_role", users_by_role)
