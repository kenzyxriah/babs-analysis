from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.models.schema import Context
from analytics.io.writers import md_table, fmt_pct, fmt_num, fmt_int, safe_label


def build_report(ctx: Context) -> None:
    r = ctx.results
    d = ctx.data

    users = d["users"]
    courses = d["courses"]
    modules = d["modules"]
    assignments = d["assignments"]
    assignment_submissions = d["assignment_submissions"]
    module_assigned_users = d["module_assigned_users"]
    payments = d["payments"]
    product_accesses = d["product_accesses"]
    user_program_selections = d.get("user_program_selections", pd.DataFrame())
    roles = d.get("roles", pd.DataFrame())

    lead_conversion = r.get("inquiry_conversion", pd.DataFrame())
    lead_tags = r.get("inquiry_intent_tags", pd.DataFrame())
    lead_volume = r.get("inquiry_volume_by_month", pd.DataFrame())
    career_goals = r.get("career_goal_buckets", pd.DataFrame())

    session_summary = r.get("session_attendance_summary", pd.DataFrame())
    join_trends = r.get("session_join_rate_trends", pd.DataFrame())
    engagement_trends = r.get("engagement_trends_over_time", pd.DataFrame())
    mau = r.get("login_monthly_active", pd.DataFrame())

    payments_status = r.get("payments_status_by_month", pd.DataFrame())
    revenue_waterfall = r.get("revenue_waterfall", pd.DataFrame())
    custom_rev = r.get("custom_product_revenue_by_month", pd.DataFrame())
    payments_received = r.get("payments_received_by_month", pd.DataFrame())
    paid_revenue_by_product = r.get("paid_revenue_by_product", pd.DataFrame())
    paid_full = r.get("paid_in_full_by_product", pd.DataFrame())
    discount_hook = r.get("discount_hook_summary", pd.DataFrame())
    payment_plan_engagement = r.get("payment_plan_engagement", pd.DataFrame())

    assignment_submission_summary = r.get("assignment_submission_summary", pd.DataFrame())
    time_to_submit = r.get("time_to_submit_distribution", pd.DataFrame())
    agreement_dist = r.get("agreement_compliance_distribution", pd.DataFrame())

    completion_breakdown = r.get("completion_threshold_breakdown", pd.DataFrame())
    ops_gaps = r.get("ops_gap_report", pd.DataFrame())
    pareto = r.get("product_revenue_pareto", pd.DataFrame())
    sales_lag = r.get("sales_lag_distribution", pd.DataFrame())
    enrollments_by_course_month = r.get("enrollments_by_course_month", pd.DataFrame())
    enrollments_by_product_month = r.get("enrollments_by_product_month", pd.DataFrame())
    product_adoption_summary = r.get("product_adoption_summary", pd.DataFrame())
    program_selection_summary = r.get("program_selection_summary", pd.DataFrame())
    specialization_tag_revenue = r.get("specialization_tag_revenue", pd.DataFrame())
    tag_category_coverage = r.get("tag_category_coverage", pd.DataFrame())

    session_attendance_rate = session_summary.copy()
    if not session_attendance_rate.empty:
        session_attendance_rate = session_attendance_rate[
            (session_attendance_rate["assignedCount"] > 0) & (session_attendance_rate["attendedCount"] > 0)
        ].copy()

    categories = d.get("categories", pd.DataFrame())
    tags = d.get("tags", pd.DataFrame())
    course_tags = d.get("course_tags", pd.DataFrame())
    product_tags = d.get("product_tags", pd.DataFrame())
    program_tags = d.get("program_tags", pd.DataFrame())

    def clean_label(value: object) -> str:
        label = safe_label(value, "", default="Unknown")
        label = "".join(ch for ch in label if ord(ch) < 128)
        label = " ".join(label.split())
        return label if label else "Unknown"

    student_role_ids = set()
    if not roles.empty and "name" in roles.columns:
        student_role_ids = set(roles[roles["name"].str.lower() == "student"]["id"].tolist())
    if not student_role_ids:
        student_role_ids = {2}
    student_ids = set(users[users["roleId"].isin(student_role_ids)]["id"].tolist()) if not users.empty else set()
    payments_filtered = payments[payments["userId"].isin(student_ids)].copy() if student_ids else payments.copy()

    lead_total = int(lead_conversion["leads"].sum()) if not lead_conversion.empty else 0
    lead_users = int(lead_conversion["users"].sum()) if not lead_conversion.empty else 0
    lead_paid = int(lead_conversion["paid_users"].sum()) if not lead_conversion.empty else 0

    pay_status_total = payments_status.groupby("status")["id"].sum().to_dict() if not payments_status.empty else {}
    pay_user_total = int(payments_filtered["userId"].nunique()) if not payments_filtered.empty else 0
    pay_users_by_status = payments_filtered.groupby("status")["userId"].nunique().to_dict() if not payments_filtered.empty else {}
    pending = float(pay_status_total.get("pending", 0))
    not_paid = float(pay_status_total.get("not_paid", 0))
    succeeded = float(pay_status_total.get("succeeded", 0))
    total_payments = pending + not_paid + succeeded
    risk_share = (pending + not_paid) / total_payments if total_payments else np.nan

    paid = payments_filtered[(payments_filtered["paidAt"].notna()) | (payments_filtered["status"].str.lower() == "succeeded")].copy()
    total_paid_revenue = float(paid["amount"].sum()) if not paid.empty else np.nan

    mau_last = int(mau.sort_values("loginMonth").iloc[-1]["MAU"]) if (not mau.empty and "MAU" in mau.columns) else 0
    mau_prev = int(mau.sort_values("loginMonth").iloc[-2]["MAU"]) if (not mau.empty and "MAU" in mau.columns and len(mau) >= 2) else 0
    mau_delta = (mau_last - mau_prev) / mau_prev if mau_prev else np.nan

    assigned_sum = session_summary["assignedCount"].sum() if not session_summary.empty else 0
    attended_sum = session_summary["attendedCount"].sum() if not session_summary.empty else 0
    overall_att_rate = attended_sum / assigned_sum if assigned_sum else np.nan
    new_face_rate = session_summary["newFaces"].sum() / attended_sum if attended_sum else np.nan

    assignment_completion_mean = float(assignment_submission_summary["submissionRate"].mean()) if not assignment_submission_summary.empty else np.nan
    median_submit_hours = float(time_to_submit["time_to_submit_hours"].median()) if not time_to_submit.empty else np.nan
    early_submit_share = float((time_to_submit["time_to_submit_hours"] < 0).mean()) if not time_to_submit.empty else np.nan

    attendance_70 = np.nan
    assignments_70 = np.nan
    strict_completion = np.nan
    if not completion_breakdown.empty and "metric" in completion_breakdown.columns:
        att_row = completion_breakdown[completion_breakdown["metric"].str.contains("attendance", case=False, na=False)]
        assn_row = completion_breakdown[completion_breakdown["metric"].str.contains("assignments", case=False, na=False)]
        strict_row = completion_breakdown[completion_breakdown["metric"].str.contains("both", case=False, na=False)]
        if not att_row.empty:
            attendance_70 = float(att_row.iloc[0]["rate"])
        if not assn_row.empty:
            assignments_70 = float(assn_row.iloc[0]["rate"])
        if not strict_row.empty:
            strict_completion = float(strict_row.iloc[0]["rate"])

    lead_peak_count = None
    lead_peak_month = None
    lead_latest_count = None
    lead_latest_month = None
    if not lead_volume.empty and "leadCount" in lead_volume.columns:
        lv = lead_volume.sort_values("leadMonth")
        peak_row = lv.loc[lv["leadCount"].idxmax()]
        latest_row = lv.iloc[-1]
        lead_peak_count = int(peak_row["leadCount"])
        lead_peak_month = pd.to_datetime(peak_row["leadMonth"])
        lead_latest_count = int(latest_row["leadCount"])
        lead_latest_month = pd.to_datetime(latest_row["leadMonth"])

    total_users = int(users["id"].nunique()) if not users.empty else 0
    conversion_rate = lead_paid / lead_total if lead_total else np.nan

    contracted_value = np.nan
    cash_collected = np.nan
    outstanding = np.nan
    if not revenue_waterfall.empty and "stage" in revenue_waterfall.columns:
        stage_map = dict(zip(revenue_waterfall["stage"].astype(str), revenue_waterfall["amount"].astype(float)))
        contracted_value = float(stage_map.get("Contracted Value", np.nan))
        cash_collected = float(stage_map.get("Cash Collected", np.nan))
        if "Outstanding" in stage_map:
            outstanding = float(stage_map.get("Outstanding", np.nan))
        elif not pd.isna(contracted_value) and not pd.isna(cash_collected):
            outstanding = float(contracted_value - cash_collected)

    completion_detail = r.get("completion_detail", pd.DataFrame())
    absconded_detail = r.get("absconded_detail", pd.DataFrame())
    course_completion_summary = r.get("course_completion_summary", pd.DataFrame())

    absconded_users = 0
    absconded_rate = np.nan
    if not absconded_detail.empty and "isAbsconded" in absconded_detail.columns:
        absconded_users = int(absconded_detail.loc[absconded_detail["isAbsconded"], "userId"].nunique())
        absconded_rate = absconded_users / total_users if total_users else np.nan

    custom_rev_total = float(custom_rev["revenue"].sum()) if not custom_rev.empty else np.nan
    payments_received_total = float(payments_received["payments"].sum()) if not payments_received.empty else np.nan

    total_categories = int(categories["name"].nunique()) if not categories.empty else 0
    total_tags = int(tags["id"].nunique()) if not tags.empty else 0
    tag_assignments = int(len(course_tags) + len(product_tags) + len(program_tags))

    program_selected_users = int(user_program_selections["userId"].nunique()) if not user_program_selections.empty else 0
    program_selection_rate = program_selected_users / total_users if total_users else np.nan
    program_major_share = np.nan
    if not user_program_selections.empty and "level" in user_program_selections.columns:
        level_counts = user_program_selections["level"].value_counts(dropna=False)
        major = float(level_counts.get("major", 0))
        total = float(level_counts.sum()) if level_counts.sum() else 0.0
        program_major_share = major / total if total else np.nan

    def fmt_money(value: float | int | None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"${float(value):,.2f}"

    def fmt_month(value: object) -> str:
        try:
            return pd.to_datetime(value).strftime("%Y-%m-%d")
        except Exception:
            return "n/a"

    peak_label = lead_peak_month.strftime("%b %Y") if lead_peak_month is not None else "n/a"
    latest_label = lead_latest_month.strftime("%b %Y") if lead_latest_month is not None else "n/a"

    mau_label = fmt_int(mau_last)
    if not pd.isna(mau_delta):
        mau_label = f"{mau_label} ({fmt_pct(mau_delta)} vs prior month)"

    lines: list[str] = []
    lines.append("# Babskenky and Company Feb 5th 2026 Report")
    lines.append("")

    lines.append("## Business Objective")
    lines.append(
        "The business objective is to lift learner activation and cash collection by improving live-session attendance, assignment completion, and paid conversion while reducing pending and not-paid exposure."
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append(
        f"Across {fmt_int(total_users)} users and {fmt_int(lead_total)} form inquiries, inquiry to paid conversion is {fmt_pct(conversion_rate)} and the latest monthly active users are {mau_label}, which frames the funnel against current active usage."
    )
    lines.append(
        f"Overall session attendance is {fmt_pct(overall_att_rate)} with a new-face rate of {fmt_pct(new_face_rate)}, while assignment completion averages {fmt_pct(assignment_completion_mean)} and the median time-to-submit is {fmt_num(median_submit_hours)} hours, which shows engagement is the primary constraint on outcomes."
    )
    lines.append(
        f"Total paid revenue is {fmt_money(total_paid_revenue)} with {fmt_pct(risk_share)} of payments pending or not paid, and the proxy absconded rate is {fmt_pct(absconded_rate)}, which makes cash collection and early activation the most urgent levers."
    )
    lines.append("")

    lines.append("## Executive KPI Summary")
    kpi_rows = [
        ["Total users", fmt_int(total_users)],
        ["Form inquiries", fmt_int(lead_total)],
        ["Latest monthly active users", mau_label],
        ["Inquiry to paid conversion", fmt_pct(conversion_rate)],
        [">=70% attendance rate", fmt_pct(attendance_70)],
        [">=70% assignment submission rate", fmt_pct(assignments_70)],
        ["Overall attendance rate", fmt_pct(overall_att_rate)],
        ["New-face rate in sessions", fmt_pct(new_face_rate)],
        ["Assignment completion (mean)", fmt_pct(assignment_completion_mean)],
        ["Median time-to-submit (hours)", fmt_num(median_submit_hours)],
        ["Total paid revenue", fmt_money(total_paid_revenue)],
        ["Pending + not paid share", fmt_pct(risk_share)],
        ["Absconded users (proxy)", f"{fmt_int(absconded_users)} ({fmt_pct(absconded_rate)})"],
    ]
    lines.append(md_table(["Metric", "Value"], kpi_rows))
    lines.append("")

    lines.append("## Glossary")
    lines.append("- Form Inquiry: A person who submitted the intake form and has not necessarily paid.")
    lines.append("- Intro Session: A short information or interview-prep session that often precedes a core course.")
    lines.append("")

    lines.append("## Learner Intent and Career Goals")
    if lead_peak_count is not None and lead_latest_count is not None:
        lines.append(
            f"Inquiry volume peaked at {fmt_int(lead_peak_count)} in {peak_label} and the latest month recorded {fmt_int(lead_latest_count)}, which shows demand timing and sensitivity to campaign cadence."
        )
    lines.append(
        f"Out of {fmt_int(lead_total)} inquiries, {fmt_int(lead_paid)} became paying users ({fmt_pct(conversion_rate)}), and the top intent tags concentrate around the primary motivations shown below, which should guide messaging and follow-up."
    )
    lines.append("")
    lines.append("![Inquiry Volume by Month](output/figures/inquiry_volume_by_month.png)")
    lines.append("![Top Inquiry Intent Tags](output/figures/inquiry_intent_tags_ranked.png)")
    lines.append("")

    if not lead_tags.empty:
        tag_top = lead_tags.sort_values("leadCount", ascending=False).head(10)
        tag_rows = [
            [clean_label(row["intentTag"]), fmt_int(row["leadCount"]), fmt_pct(row.get("share"))]
            for _, row in tag_top.iterrows()
        ]
        lines.append("### Top Inquiry Intent Tags (Top 10)")
        lines.append(md_table(["Intent Tag", "Inquiries", "Share"], tag_rows))
        lines.append("")

    if not career_goals.empty:
        goal_rows = [
            [clean_label(row["goalBucket"]), fmt_int(row["count"]), fmt_pct(row.get("share"))]
            for _, row in career_goals.sort_values("count", ascending=False).iterrows()
        ]
        lines.append("### Career Goal Buckets")
        lines.append(md_table(["Goal Bucket", "Count", "Share"], goal_rows))
        lines.append("")
        lines.append("![Career Goal Distribution (Top Buckets)](output/figures/career_goal_distribution.png)")
        lines.append("")

    lines.append("## Attendance and Session Engagement")
    total_sessions = (
        int(session_summary.loc[session_summary["assignedCount"] > 0, "liveSessionId"].nunique())
        if not session_summary.empty and "liveSessionId" in session_summary.columns
        else 0
    )
    lines.append(
        f"Across {fmt_int(total_sessions)} sessions, the overall attendance rate is {fmt_pct(overall_att_rate)} and the new-face rate is {fmt_pct(new_face_rate)}, which indicates that sessions are under-attended and growth is driven more by repeat attendees than fresh attendance."
    )
    lines.append(
        "The lowest attendance sessions identify the clearest near-term improvement targets, so tightening reminders and repositioning those sessions should lift overall attendance fastest."
    )
    lines.append("")

    if not session_attendance_rate.empty:
        low_sessions = session_attendance_rate.sort_values("joinRate").head(10)
        low_rows = []
        for _, row in low_sessions.iterrows():
            low_rows.append(
                [
                    clean_label(row.get("sessionTitle")),
                    fmt_month(row.get("scheduledAt")),
                    fmt_int(row.get("assignedCount")),
                    fmt_int(row.get("attendedCount")),
                    fmt_pct(row.get("joinRate")),
                    fmt_pct(row.get("newFaceRate")),
                ]
            )
        lines.append("### Lowest Attendance Rate Sessions (Top 10)")
        lines.append(md_table(["Session", "Date", "Assigned", "Attended", "Attendance Rate", "New-Face Rate"], low_rows))
        lines.append("")

    lines.append("![Attendance Rate Distribution (Sessions)](output/figures/attendance_rate_distribution.png)")
    lines.append("")
    lines.append("![Session Join Rate Over Time](output/figures/session_join_rate_trends.png)")
    lines.append("")

    lines.append("## Assignment Engagement and Submission Speed")
    lines.append(
        f"Average assignment completion is {fmt_pct(assignment_completion_mean)} and the median submission timing is {fmt_num(median_submit_hours)} hours from the baseline, while {fmt_pct(early_submit_share)} of submissions arrive before the baseline, which shows pacing varies and deadlines are not the dominant driver."
    )
    lines.append("")

    if not assignment_submission_summary.empty:
        assn_top = assignment_submission_summary.sort_values("submissions", ascending=False).head(10)
        assn_rows = [
            [clean_label(row.get("title")), fmt_int(row.get("submissions")), fmt_pct(row.get("submissionRate"))]
            for _, row in assn_top.iterrows()
        ]
        lines.append("### Assignment Completion vs Active Students (Top 10)")
        lines.append(md_table(["Assignment", "Submitted", "Completion vs Active"], assn_rows))
        lines.append("")

    lines.append("![Time-to-Submit Distribution (Hours)](output/figures/time_to_submit_distribution.png)")
    lines.append("")

    lines.append("## Revenue and Payment Health")
    lines.append(
        f"Custom product revenue totals {fmt_money(custom_rev_total)} and paid receipts total {fmt_money(payments_received_total)}, which indicates that booked value is converting to cash but with variability across months."
    )
    lines.append(
        f"Pending and not paid payments represent {fmt_pct(risk_share)} of all transactions, which means collections remain a core operational risk alongside engagement."
    )
    lines.append("")

    if pay_status_total:
        status_rows = []
        for status in ["succeeded", "pending", "not_paid"]:
            if status in pay_status_total:
                status_rows.append(
                    [
                        status,
                        fmt_int(pay_status_total.get(status)),
                        fmt_int(pay_users_by_status.get(status)),
                    ]
                )
        lines.append("### Payment Status Summary")
        lines.append(md_table(["Status", "Payment Records", "Unique Users"], status_rows))
        lines.append("")
        if pay_user_total:
            lines.append(
                f"Payment status counts are based on payment records for Student users only, and {fmt_int(pay_user_total)} unique students have at least one payment record in the system."
            )
            lines.append("")

    lines.append("![Payments Status by Month](output/figures/payments_status_by_month.png)")
    lines.append("")

    if not revenue_waterfall.empty:
        lines.append("![Revenue Waterfall](output/figures/revenue_waterfall.png)")
        lines.append("")
        if not pd.isna(contracted_value) and not pd.isna(cash_collected):
            lines.append(
                f"Contracted value is {fmt_money(contracted_value)}, cash collected is {fmt_money(cash_collected)}, and outstanding value is {fmt_money(outstanding)}, which makes the booked versus collected gap visible at a glance."
            )
            lines.append("")

    if not custom_rev.empty:
        rows = [[fmt_month(row["revenueMonth"]), fmt_money(row["revenue"])] for _, row in custom_rev.sort_values("revenueMonth").iterrows()]
        lines.append("### Monthly Custom Product Revenue")
        lines.append(md_table(["Month", "Revenue"], rows))
        lines.append("")
        lines.append("![Custom Product Revenue by Month](output/figures/custom_product_revenue_by_month.png)")
        lines.append("")

    if not payments_received.empty:
        rows = [[fmt_month(row["paidMonth"]), fmt_money(row["payments"])] for _, row in payments_received.sort_values("paidMonth").iterrows()]
        lines.append("### Monthly Payments Received")
        lines.append(md_table(["Month", "Payments Received"], rows))
        lines.append("")
        lines.append("![Payments Received by Month](output/figures/payments_received_by_month.png)")
        lines.append("")

    if not paid_revenue_by_product.empty:
        paid_rev = paid_revenue_by_product.copy()
        if not paid_full.empty and "paid_in_full_rate" in paid_full.columns:
            paid_rev = paid_rev.merge(paid_full[["productId", "paid_in_full_rate"]], on="productId", how="left")
        paid_rev = paid_rev.sort_values("paidRevenue", ascending=False).head(10)
        paid_rows = [
            [
                clean_label(row.get("productTitle", row.get("productId"))),
                fmt_money(row.get("paidRevenue")),
                fmt_pct(row.get("paid_in_full_rate")),
            ]
            for _, row in paid_rev.iterrows()
        ]
        lines.append("### Top Products by Paid Revenue")
        lines.append(md_table(["Product", "Paid Revenue", "Fully Paid Rate"], paid_rows))
        lines.append("")

    if not discount_hook.empty:
        disc = discount_hook.sort_values("discount_share", ascending=False).head(10)
        disc_rows = [
            [
                clean_label(row.get("productTitle", row.get("productId"))),
                fmt_int(row.get("discount_sales")),
                fmt_int(row.get("full_sales")),
                fmt_int(row.get("total_sales")),
                fmt_pct(row.get("discount_share")),
            ]
            for _, row in disc.iterrows()
        ]
        lines.append("### Discount Usage Summary")
        lines.append(md_table(["Product", "Discount Sales", "Full Sales", "Total Sales", "Discount Share"], disc_rows))
        lines.append("")
        if not disc.empty:
            top_disc = disc.iloc[0]
            lines.append(
                f"{clean_label(top_disc.get('productTitle', top_disc.get('productId')))} has the highest discount share at {fmt_pct(top_disc.get('discount_share'))}, which signals pricing sensitivity or a need to sharpen value framing."
            )
            lines.append("")

    if not payment_plan_engagement.empty:
        plan_rows = [
            ["Installment" if bool(row.get("is_installment")) else "Full pay", fmt_int(row.get("users")), fmt_num(row.get("avg_submissions"))]
            for _, row in payment_plan_engagement.iterrows()
        ]
        lines.append("### Payment Plan Engagement")
        lines.append(md_table(["Plan", "Users", "Avg Submissions"], plan_rows))
        lines.append("")

    lines.append("## Agreement Compliance")
    if not agreement_dist.empty and "hoursToAgree" in agreement_dist.columns:
        hours = agreement_dist["hoursToAgree"].dropna()
        if not hours.empty:
            agreement_rows = [
                ["Mean hours to agree", fmt_num(hours.mean())],
                ["Median hours to agree", fmt_num(hours.median())],
                ["75th percentile hours", fmt_num(hours.quantile(0.75))],
                ["Max hours to agree", fmt_num(hours.max())],
            ]
            lines.append(
                f"The median agreement time is {fmt_num(hours.median())} hours and the 75th percentile is {fmt_num(hours.quantile(0.75))} hours, which means a meaningful share of learners take multiple days to accept agreements and this can delay progress."
            )
            lines.append("")
            lines.append("### Agreement Compliance Summary")
            lines.append(md_table(["Metric", "Value"], agreement_rows))
            lines.append("")
            lines.append("![Agreement Compliance Time (Hours)](output/figures/agreement_compliance_time.png)")
            lines.append("")

    lines.append("## Learning Engagement (Attendance + Work Submissions)")
    if not mau.empty:
        lines.append(
            f"Monthly active users ended at {fmt_int(mau_last)}, which is {fmt_pct(mau_delta)} versus the prior month, and sustained dips at this level would signal weaker engagement or missed reminders."
        )
    if not engagement_trends.empty and len(engagement_trends) >= 6:
        trends = engagement_trends.sort_values("month")
        last3 = trends.tail(3)
        prev3 = trends.iloc[-6:-3]
        att_last = last3["attendanceEvents"].mean()
        att_prev = prev3["attendanceEvents"].mean()
        sub_last = last3["submissionEvents"].mean()
        sub_prev = prev3["submissionEvents"].mean()
        att_change = (att_last - att_prev) / att_prev if att_prev else np.nan
        sub_change = (sub_last - sub_prev) / sub_prev if sub_prev else np.nan
        lines.append(
            f"The last three months average {fmt_num(att_last)} attendance events and {fmt_num(sub_last)} submission events, which is {fmt_pct(att_change)} and {fmt_pct(sub_change)} versus the prior three months, showing whether engagement is rising or falling together."
        )
    lines.append("")
    lines.append("![Monthly Active Users](output/figures/monthly_active_users.png)")
    lines.append("![Engagement Trends Over Time](output/figures/engagement_trends_over_time.png)")
    lines.append("")

    if not completion_breakdown.empty:
        comp_rows = [
            [clean_label(row.get("metric")), fmt_int(row.get("users")), fmt_pct(row.get("rate"))]
            for _, row in completion_breakdown.iterrows()
        ]
        lines.append("### Completion Threshold Breakdown")
        lines.append(md_table(["Completion Threshold", "Users", "Rate"], comp_rows))
        lines.append("")

    if not course_completion_summary.empty:
        lines.append("![Completion Rate by Course](output/figures/completion_rate_by_course.png)")
        lines.append("")

    lines.append("## Program and Catalog Strategy")
    lines.append(
        f"The updated catalog taxonomy now spans {fmt_int(total_categories)} category types and {fmt_int(total_tags)} tags, with {fmt_int(tag_assignments)} total tag assignments across courses, products, and programs, which provides a clearer structure for discovery and marketing."
    )
    lines.append(
        f"Program selections cover {fmt_int(program_selected_users)} users ({fmt_pct(program_selection_rate)} of the user base), and majors make up {fmt_pct(program_major_share)} of selections, which indicates how focused learners are in their chosen pathways."
    )
    lines.append("")

    if not program_selection_summary.empty:
        prog_top = program_selection_summary.sort_values("selected_users", ascending=False).head(10)
        prog_rows = [
            [
                clean_label(row.get("programTitle", row.get("programId"))),
                fmt_int(row.get("selected_users")),
                fmt_pct(row.get("major_share")),
                fmt_int(row.get("linked_courses")),
                fmt_int(row.get("linked_products")),
            ]
            for _, row in prog_top.iterrows()
        ]
        lines.append("### Program Selection Summary (Top 10)")
        lines.append(md_table(["Program", "Selected Users", "Major Share", "Linked Courses", "Linked Products"], prog_rows))
        lines.append("")

    if not specialization_tag_revenue.empty:
        spec_top = specialization_tag_revenue.sort_values("attributed_paid_revenue", ascending=False).head(10)
        spec_rows = [
            [
                clean_label(row.get("tag")),
                fmt_money(row.get("attributed_paid_revenue")),
                fmt_int(row.get("tagged_products")),
            ]
            for _, row in spec_top.iterrows()
        ]
        lines.append("### Specialization Tags by Attributed Paid Revenue (Top 10)")
        lines.append(md_table(["Specialization Tag", "Attributed Paid Revenue", "Tagged Products"], spec_rows))
        lines.append("")
        if not spec_top.empty:
            top_tag = spec_top.iloc[0]
            lines.append(
                f"{clean_label(top_tag.get('tag'))} leads specialization-linked paid revenue at {fmt_money(top_tag.get('attributed_paid_revenue'))}, which helps prioritize where revenue-driven positioning is strongest."
            )
            lines.append("")

    if tag_category_coverage is not None and not tag_category_coverage.empty:
        lines.append("![Tag Category Coverage by Entity Type](output/figures/tag_category_coverage.png)")
        lines.append("")

    lines.append("## Course Completion and Product Adoption")
    if not completion_detail.empty:
        assignments_with_course = assignments.merge(
            modules[["id", "courseId"]],
            left_on="moduleId",
            right_on="id",
            how="left",
            suffixes=("", "_module"),
        )
        assignment_counts = assignments_with_course.groupby("courseId")["id"].nunique().reset_index().rename(columns={"id": "assignments"})
        completion_proxy = completion_detail.groupby("courseId").agg(
            assigned_users=("userId", "nunique"),
            any_submission=("assignmentCompletionRate", lambda s: (s > 0).sum()),
            all_assignments=("assignmentCompletionRate", lambda s: (s >= 0.999).sum()),
            any_rate=("assignmentCompletionRate", lambda s: (s > 0).mean()),
            all_rate=("assignmentCompletionRate", lambda s: (s >= 0.999).mean()),
        ).reset_index()
        completion_proxy = completion_proxy.merge(assignment_counts, on="courseId", how="left")
        completion_proxy = completion_proxy[completion_proxy["assignments"].fillna(0) > 0]
        completion_proxy = completion_proxy.merge(courses[["id", "title"]], left_on="courseId", right_on="id", how="left")
        completion_proxy["courseTitle"] = completion_proxy["title"].fillna(completion_proxy["courseId"])
        completion_proxy = completion_proxy.sort_values("assigned_users", ascending=False).head(10)
        proxy_rows = [
            [
                clean_label(row.get("courseTitle"))[:45],
                fmt_int(row.get("assigned_users")),
                fmt_int(row.get("assignments")),
                fmt_int(row.get("any_submission")),
                fmt_int(row.get("all_assignments")),
                fmt_pct(row.get("any_rate")),
                fmt_pct(row.get("all_rate")),
            ]
            for _, row in completion_proxy.iterrows()
        ]
        lines.append("### Course Completion Summary (Proxy)")
        lines.append(
            md_table(
                ["Course", "Assigned Users", "Assignments", "Any Submission", "All Assignments", "Any Rate", "All Rate"],
                proxy_rows,
            )
        )
        lines.append("")

    if not product_adoption_summary.empty:
        adoption_top = product_adoption_summary.sort_values("unique_users", ascending=False).head(10)
        adoption_rows = [
            [
                clean_label(row.get("productTitle", row.get("productId"))),
                fmt_int(row.get("unique_users")),
                fmt_int(row.get("active_users")),
                fmt_pct(row.get("adoption_rate")),
            ]
            for _, row in adoption_top.iterrows()
        ]
        lines.append("### Product Adoption (Top 10)")
        lines.append(md_table(["Product", "Unique Users", "Active Users", "Adoption Rate"], adoption_rows))
        lines.append("")
        lines.append("![Product Adoption (Top 12)](output/figures/product_adoption_top12.png)")
        lines.append("")

    lines.append("## Product Strategy and Enrollments")
    if not enrollments_by_course_month.empty:
        course_totals = enrollments_by_course_month.groupby("courseTitle")["userId"].sum().sort_values(ascending=False)
        course_share = course_totals.head(5).sum() / course_totals.sum() if course_totals.sum() else np.nan
        lines.append(
            f"The top five courses account for {fmt_pct(course_share)} of enrollments, so improving onboarding and retention in those courses will move outcomes fastest."
        )
    if not enrollments_by_product_month.empty:
        product_totals = enrollments_by_product_month.groupby("productTitle")["userId"].sum().sort_values(ascending=False)
        product_share = product_totals.head(5).sum() / product_totals.sum() if product_totals.sum() else np.nan
        lines.append(
            f"The top five products account for {fmt_pct(product_share)} of enrollments, which reinforces the focus on a small set of offerings." 
        )
    lines.append("")
    lines.append("![Top-5 Course Enrollments](output/figures/enrollments_top5_courses.png)")
    lines.append("![Top-5 Product Enrollments](output/figures/enrollments_top5_products.png)")
    lines.append("")

    if not pareto.empty:
        lines.append("![Revenue Concentration (Pareto)](output/figures/product_revenue_pareto.png)")
        lines.append("")
        if "cumulative_share" in pareto.columns:
            try:
                k = int((pareto["cumulative_share"] >= 0.8).idxmax()) + 1
                lines.append(
                    f"Roughly {fmt_int(k)} products drive about 80% of revenue, which means protecting and improving these products is the fastest lever for revenue stability."
                )
            except Exception:
                lines.append(
                    "Revenue is concentrated in a small number of products, which means protecting those products should be a priority for stability."
                )
            lines.append("")

    lines.append("## Sales Velocity")
    if not sales_lag.empty and "salesLagDays" in sales_lag.columns:
        lag = sales_lag["salesLagDays"].dropna()
        if not lag.empty:
            lines.append(
                f"Average sales lag is {fmt_num(lag.mean())} days and the median is {fmt_num(lag.median())} days, which shows that the form-to-payment cycle remains longer than ideal."
            )
            lines.append("")
            lines.append("![Sales Lag (Histogram)](output/figures/sales_lag_hist.png)")
            lines.append("")

    lines.append("## Revenue Leakage and Operational Risks")
    if not ops_gaps.empty:
        gap_df = ops_gaps.rename(columns={"gapType": "gap", "userCount": "users", "notes": "meaning"})
        gap_rows = [
            [clean_label(row.get("gap")), fmt_int(row.get("users")), clean_label(row.get("meaning"))]
            for _, row in gap_df.iterrows()
        ]
        lines.append(md_table(["Gap", "Users", "Meaning"], gap_rows))
        lines.append("")
        if "users" in gap_df.columns and not gap_df.empty:
            top_gap = gap_df.sort_values("users", ascending=False).iloc[0]
            lines.append(
                f"The largest operational gap is {clean_label(top_gap.get('gap'))} affecting {fmt_int(top_gap.get('users'))} users, which signals immediate leakage risk that can be addressed with tighter enrollment and payment reconciliation."
            )
            lines.append("")
        lines.append(
            "Exceptions and mismatched access/payment are direct revenue leakage, so tightening these controls is the fastest way to reduce losses without changing the product."
        )
        lines.append("")

    lines.append("## Relationships and Drivers")
    lines.append(
        "Sales velocity, early activation, and session engagement quality appear tightly linked, because longer form-to-payment lag reduces conversion, weaker attendance and submissions reduce completion, and low new-face rates indicate recycling the same attendees rather than expanding reach."
    )
    lines.append(
        "Improving any one of these areas tends to lift the others, so faster follow-up, stronger onboarding, and clearer session positioning should collectively raise conversion, engagement, and cash collection."
    )
    lines.append("")

    lines.append("## Summary and Overall Health")
    lines.append(
        f"Business health is mixed, with {fmt_pct(risk_share)} pending or not paid share, strict completion at {fmt_pct(strict_completion)}, and an absconded rate of {fmt_pct(absconded_rate)}, which means revenue is being booked without full collection and many learners disengage before completing."
    )
    lines.append(
        "Coverage is strongest for payments, sessions, and inquiries, while strict completion depends on both attendance and assignment data and should be read as a lower bound until validated against manual completion records."
    )
    lines.append(
        "The business is generating interest but losing momentum across sales velocity and early engagement, so tightening follow-up speed, improving session activation, and enforcing payment compliance should lift conversion, cash collection, and completion together."
    )
    lines.append("")

    report_path = ctx.settings.base_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
