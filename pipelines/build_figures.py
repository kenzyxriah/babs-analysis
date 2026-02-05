from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from analytics.models.schema import Context
from analytics.io.writers import save_fig
from analytics.visuals.style import custom_theme, annotate_point, add_headroom, PALETTE, VIBRANT_COLORS


def build_figures(ctx: Context) -> None:
    fig_dir = ctx.settings.fig_dir
    with plt.style.context(custom_theme()):
        _build_figures(ctx, fig_dir)


def _build_figures(ctx: Context, fig_dir) -> None:

    # 1) Top-5 course enrollments (no "Other")
    df = ctx.results.get("enrollments_by_course_month")
    if df is not None and not df.empty:
        label_col = "courseTitle" if "courseTitle" in df.columns else "courseId"
        totals = df.groupby(label_col)["userId"].sum().sort_values(ascending=False)
        top5 = totals.head(5).index.tolist()
        plot_df = df[df[label_col].isin(top5)].copy()
        plot_df["enrollmentMonth"] = pd.to_datetime(plot_df["enrollmentMonth"])

        fig, ax = plt.subplots(figsize=(8, 4))
        line_colors = [
            PALETTE["primary"],
            PALETTE["secondary"],
            VIBRANT_COLORS[0],
            VIBRANT_COLORS[1],
            VIBRANT_COLORS[2],
        ]
        for idx, (course_id, group) in enumerate(plot_df.groupby(label_col)):
            label = str(course_id)[:24]
            color = line_colors[min(idx, len(line_colors) - 1)]
            ax.plot(group["enrollmentMonth"], group["userId"], label=label, color=color)

        top_course = totals.index[0]
        top_series = plot_df[plot_df[label_col] == top_course]
        if not top_series.empty:
            peak_row = top_series.loc[top_series["userId"].idxmax()]
            annotate_point(ax, "Peak", (peak_row["enrollmentMonth"], peak_row["userId"]))

        ax.set_title("Top-5 Course Enrollments")
        ax.set_xlabel("Month")
        ax.set_ylabel("Enrollments")
        ax.legend(fontsize=8, ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "enrollments_top5_courses.png")

    # 1b) Top-5 product enrollments (no "Other")
    df = ctx.results.get("enrollments_by_product_month")
    if df is not None and not df.empty:
        label_col = "productTitle" if "productTitle" in df.columns else "productId"
        totals = df.groupby(label_col)["userId"].sum().sort_values(ascending=False)
        top5 = totals.head(5).index.tolist()
        plot_df = df[df[label_col].isin(top5)].copy()
        plot_df["enrollmentMonth"] = pd.to_datetime(plot_df["enrollmentMonth"])

        fig, ax = plt.subplots(figsize=(8, 4))
        line_colors = [
            PALETTE["primary"],
            PALETTE["secondary"],
            VIBRANT_COLORS[0],
            VIBRANT_COLORS[1],
            VIBRANT_COLORS[2],
        ]
        for idx, (product_id, group) in enumerate(plot_df.groupby(label_col)):
            label = str(product_id)[:24]
            color = line_colors[min(idx, len(line_colors) - 1)]
            ax.plot(group["enrollmentMonth"], group["userId"], label=label, color=color)

        top_product = totals.index[0]
        top_series = plot_df[plot_df[label_col] == top_product]
        if not top_series.empty:
            peak_row = top_series.loc[top_series["userId"].idxmax()]
            annotate_point(ax, "Peak", (peak_row["enrollmentMonth"], peak_row["userId"]))

        ax.set_title("Top-5 Product Enrollments")
        ax.set_xlabel("Month")
        ax.set_ylabel("Enrollments")
        ax.legend(fontsize=8, ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "enrollments_top5_products.png")

    # 2) Payments status by month with callouts
    df = ctx.results.get("payments_status_by_month")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot = df.pivot(index="paymentMonth", columns="status", values="id").fillna(0)
        pivot.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE["primary"], PALETTE["secondary"], VIBRANT_COLORS[0]])
        ax.set_title("Payments Status by Month", pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")

        if "pending" in pivot.columns:
            peak_idx = pivot["pending"].idxmax()
            peak_val = pivot.loc[peak_idx, "pending"]
            annotate_point(ax, f"Peak pending: {int(peak_val)}", (pivot.index.get_loc(peak_idx), pivot.loc[peak_idx].sum()))

        ax.set_xticklabels([pd.to_datetime(x).strftime("%b %Y") for x in pivot.index], rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "payments_status_by_month.png")

    # 2b) Custom product revenue by month
    df = ctx.results.get("custom_product_revenue_by_month")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        df = df.sort_values("revenueMonth")
        ax.plot(df["revenueMonth"], df["revenue"], marker="o", color=PALETTE["primary"])
        ax.set_title("Custom Product Revenue by Month", pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "custom_product_revenue_by_month.png")

    # 2c) Payments received by month
    df = ctx.results.get("payments_received_by_month")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        df = df.sort_values("paidMonth")
        ax.plot(df["paidMonth"], df["payments"], marker="o", color=PALETTE["secondary"])
        ax.set_title("Payments Received by Month", pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Payments")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "payments_received_by_month.png")

    # 3) Completion rate by course with threshold note
    df = ctx.results.get("course_completion_summary")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = df["courseTitle"].fillna(df["courseId"]).astype(str).str[:12]
        ax.bar(labels, df["completionRate"], color=PALETTE["primary"])
        ax.set_title("Completion Rate by Course", pad=12)
        ax.set_xlabel("Course")
        ax.set_ylabel("Completion Rate")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.text(0.01, 0.95, "Threshold: >=70% attendance & >=70% assignments", transform=ax.transAxes, fontsize=9)
        add_headroom(ax)
        save_fig(fig, fig_dir / "completion_rate_by_course.png")

    # 3b) Product adoption (Top 12)
    df = ctx.results.get("product_adoption_summary")
    if df is not None and not df.empty:
        df = df.sort_values("unique_users", ascending=False).head(12)
        labels = df["productTitle"].fillna(df["productId"]).astype(str)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, df["unique_users"], color=PALETTE["primary"])
        ax.set_title("Product Adoption (Top 12)", pad=12)
        ax.set_xlabel("Unique Users")
        for i, v in enumerate(df["unique_users"]):
            ax.text(v + 0.5, i, str(int(v)), va="center", fontsize=10)
        save_fig(fig, fig_dir / "product_adoption_top12.png")

    # 4) Lead volume with MoM labels
    df = ctx.results.get("inquiry_volume_by_month")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        df = df.sort_values("leadMonth")
        ax.plot(df["leadMonth"], df["leadCount"], marker="o", color=PALETTE["primary"])
        ax.set_title("Inquiry Volume by Month", pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Inquiries")

        peak_row = df.loc[df["leadCount"].idxmax()]
        annotate_point(ax, f"Peak: {int(peak_row['leadCount'])}", (peak_row["leadMonth"], peak_row["leadCount"]))

        # Light MoM callouts (skip first month)
        try:
            mom = df["leadCount"].pct_change()
            for i in range(1, len(df)):
                if pd.isna(mom.iloc[i]):
                    continue
                ax.text(
                    df.iloc[i]["leadMonth"],
                    df.iloc[i]["leadCount"] + 0.4,
                    f"{mom.iloc[i]:+.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=PALETTE["neutral"],
                )
        except Exception:
            pass

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "inquiry_volume_by_month.png")

    # 5) Lead intent tags ranked horizontal
    df = ctx.results.get("inquiry_intent_tags")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        top_tags = df.sort_values("leadCount", ascending=False).head(10)
        ax.barh(top_tags["intentTag"], top_tags["leadCount"], color=PALETTE["primary"])
        ax.invert_yaxis()
        ax.set_title("Top Inquiry Intent Tags", pad=12)
        ax.set_xlabel("Inquiries")
        ax.set_ylabel("Intent Tag")
        max_v = float(top_tags["leadCount"].max()) if not top_tags.empty else 0.0
        ax.set_xlim(0, max_v * 1.25 if max_v else 1)
        for i, v in enumerate(top_tags["leadCount"]):
            ax.text(v + (max_v * 0.03 if max_v else 0.5), i, str(int(v)), va="center", fontsize=12, color="black")
        save_fig(fig, fig_dir / "inquiry_intent_tags_ranked.png")

    # 5b) Career goal distribution
    df = ctx.results.get("career_goal_buckets")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        top_goals = df.sort_values("count", ascending=False).head(8)
        ax.barh(top_goals["goalBucket"].astype(str), top_goals["count"], color=PALETTE["primary"])
        ax.invert_yaxis()
        ax.set_title("Career Goal Distribution (Top Buckets)", pad=12)
        ax.set_xlabel("Submissions")
        for i, v in enumerate(top_goals["count"]):
            ax.text(v + 0.5, i, str(int(v)), va="center", fontsize=12, color="black")
        save_fig(fig, fig_dir / "career_goal_distribution.png")

    # 5c) Tag category coverage by entity type
    df = ctx.results.get("tag_category_coverage")
    if df is not None and not df.empty:
        pivot = df.pivot(index="category", columns="entityType", values="entities").fillna(0)
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot.plot(kind="bar", ax=ax, color=[PALETTE["primary"], PALETTE["secondary"], VIBRANT_COLORS[0]])
        ax.set_title("Tag Category Coverage by Entity Type", pad=12)
        ax.set_xlabel("Category")
        ax.set_ylabel("Tagged Entities")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "tag_category_coverage.png")

    # 6) Session attendance vs assigned with diagonal
    df = ctx.results.get("session_attendance_summary")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df["assignedCount"], df["attendedCount"], alpha=0.6, color=PALETTE["primary"])
        max_val = max(df["assignedCount"].max(), df["attendedCount"].max())
        ax.plot([0, max_val], [0, max_val], linestyle="--", color=PALETTE["neutral"])
        ax.set_title("Session Attendance: Assigned vs Attended", pad=12)
        ax.set_xlabel("Assigned")
        ax.set_ylabel("Attended")

        worst = df.sort_values("joinRate").head(3)
        for _, row in worst.iterrows():
            annotate_point(ax, "Low join", (row["assignedCount"], row["attendedCount"]))

        add_headroom(ax)
        save_fig(fig, fig_dir / "session_attendance_vs_assigned.png")

        attended_only = df[(df["attendedCount"] > 0) & (df["assignedCount"] > 0)].copy()
        low = attended_only.sort_values("joinRate").head(10) if not attended_only.empty else df.sort_values("joinRate").head(10)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(low["sessionTitle"].fillna("Unknown"), low["joinRate"] * 100, color=PALETTE["primary"])
        ax.set_title("Lowest Attendance Rate Sessions (Top 10)", pad=12)
        ax.set_xlabel("Attendance Rate (%)")
        for i, v in enumerate(low["joinRate"] * 100):
            if pd.notna(v):
                ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
        save_fig(fig, fig_dir / "attendance_rate_lowest_top10.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        hist_vals = attended_only["joinRate"].dropna() * 100 if not attended_only.empty else df["joinRate"].dropna() * 100
        ax.hist(hist_vals, bins=10, color=PALETTE["secondary"], alpha=0.8)
        ax.set_title("Attendance Rate Distribution (Sessions)", pad=12)
        ax.set_xlabel("Attendance Rate (%)")
        ax.set_ylabel("Session Count")
        save_fig(fig, fig_dir / "attendance_rate_distribution.png")

    # 7) Join rate trend (Intro vs Core)
    df = ctx.results.get("session_join_rate_trends")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        label_map = {
            "gateway": "Intro Session",
            "intro": "Intro Session",
            "non_gateway": "Core Session",
            "non-gateway": "Core Session",
            "core": "Core Session",
        }
        series_colors = [PALETTE["primary"], PALETTE["secondary"]]
        for idx, (label, group) in enumerate(df.groupby("sessionType")):
            key = str(label).strip().lower()
            pretty = label_map.get(key, label)
            color = series_colors[min(idx, len(series_colors) - 1)]
            ax.plot(group["sessionMonth"], group["avg_join_rate"], marker="o", label=pretty, color=color)
        ax.set_title("Session Join Rate Over Time (Intro vs Core)", pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Avg Join Rate")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "session_join_rate_trends.png")

    # 8) Engagement trends over time (attendance + submissions)
    df = ctx.results.get("engagement_trends_over_time")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["month"], df["attendanceEvents"], marker="o", color=PALETTE["primary"], label="Attendance")
        ax.plot(df["month"], df["submissionEvents"], marker="o", color=PALETTE["secondary"], label="Submissions")
        ax.set_title("Engagement Trends Over Time", pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Events")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "engagement_trends_over_time.png")

    # 10) Revenue waterfall (simple)
    df = ctx.results.get("revenue_waterfall")
    if df is not None and not df.empty:
        # True waterfall: Contracted (start) -> subtract Cash -> Ending Outstanding.
        fig, ax = plt.subplots(figsize=(7, 4))
        stage_map = dict(zip(df["stage"].astype(str), df["amount"].astype(float)))
        contracted = float(stage_map.get("Contracted Value", 0.0))
        cash = float(stage_map.get("Cash Collected", 0.0))
        outstanding = float(stage_map.get("Outstanding", max(contracted - cash, 0.0)))

        stages = ["Contracted Value", "Cash Collected", "Outstanding"]
        # Start, change, end
        heights = [contracted, -cash, outstanding]
        bottoms = [0.0, contracted, 0.0]
        colors_ = [PALETTE["primary"], PALETTE["secondary"], VIBRANT_COLORS[0]]
        ax.bar(stages, heights, bottom=bottoms, color=colors_)

        # Connector line from end of contracted to end of remaining after cash
        remaining = contracted - cash
        ax.plot([0, 1], [contracted, remaining], linestyle="--", color=PALETTE["neutral"], linewidth=1)

        ax.set_title("Revenue Waterfall: Contracted vs Collected vs Outstanding", pad=12)
        ax.set_ylabel("Amount")

        ax.text(0, contracted, f"{contracted:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(1, contracted - cash, f"-{cash:,.0f}", ha="center", va="top", fontsize=9)
        ax.text(2, outstanding, f"{outstanding:,.0f}", ha="center", va="bottom", fontsize=9)
        add_headroom(ax)
        save_fig(fig, fig_dir / "revenue_waterfall.png")

    # 10b) Agreement compliance time distribution
    df = ctx.results.get("agreement_compliance_distribution")
    if df is not None and not df.empty and "hoursToAgree" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["hoursToAgree"].dropna(), bins=20, color=PALETTE["primary"], alpha=0.8)
        ax.set_title("Agreement Compliance Time (Hours)", pad=12)
        ax.set_xlabel("Hours to Agree")
        ax.set_ylabel("Count")
        save_fig(fig, fig_dir / "agreement_compliance_time.png")

    # 12) Revenue Pareto with 80% reference
    df = ctx.results.get("product_revenue_pareto")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["cumulative_share"], marker="o", color=PALETTE["primary"])
        ax.axhline(0.8, color="#C00000", linestyle="--")
        ax.set_title("Revenue Concentration (Pareto)", pad=12)
        ax.set_xlabel("Product Rank")
        ax.set_ylabel("Cumulative Revenue Share")
        # Annotate how many products reach 80% revenue
        try:
            k = int((df["cumulative_share"] >= 0.8).idxmax()) + 1
            annotate_point(ax, f"80% reached by ~{k} products", (k - 1, float(df.iloc[k - 1]["cumulative_share"])))
        except Exception:
            annotate_point(ax, "80% line", (0, 0.8))
        save_fig(fig, fig_dir / "product_revenue_pareto.png")

    # 14) Sales lag histogram with mean/median
    df = ctx.results.get("sales_lag_distribution")
    if df is not None and not df.empty and "salesLagDays" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = df["salesLagDays"].dropna()
        ax.hist(vals, bins=15, color=PALETTE["primary"], alpha=0.7)
        mean = vals.mean()
        median = vals.median()
        ax.axvline(mean, color="#C00000", linestyle="--", label=f"Mean: {mean:.1f}")
        ax.axvline(median, color="#0057B8", linestyle="--", label=f"Median: {median:.1f}")
        ax.set_title("Sales Lag (Days)")
        ax.set_xlabel("Days")
        ax.set_ylabel("Count")
        ax.legend()
        save_fig(fig, fig_dir / "sales_lag_hist.png")

    # 15) Time-to-submit distribution (hours)
    df = ctx.results.get("time_to_submit_distribution")
    if df is not None and not df.empty and "time_to_submit_hours" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["time_to_submit_hours"].dropna(), bins=20, color=PALETTE["primary"], alpha=0.8)
        ax.set_title("Time-to-Submit Distribution (Hours)", pad=12)
        ax.set_xlabel("Hours from Baseline")
        ax.set_ylabel("Count")
        save_fig(fig, fig_dir / "time_to_submit_distribution.png")

    # 16) Monthly active users with 3-month average
    df = ctx.results.get("login_monthly_active")
    if df is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        df = df.sort_values("loginMonth")
        ax.plot(df["loginMonth"], df["MAU"], marker="o", color=PALETTE["primary"], label="MAU")
        df["MAU_MA3"] = df["MAU"].rolling(3).mean()
        ax.plot(df["loginMonth"], df["MAU_MA3"], color=PALETTE["secondary"], label="3-Month Avg")
        peak = df.loc[df["MAU"].idxmax()]
        trough = df.loc[df["MAU"].idxmin()]
        annotate_point(ax, "Peak", (peak["loginMonth"], peak["MAU"]))
        annotate_point(ax, "Trough", (trough["loginMonth"], trough["MAU"]))
        ax.set_title("Monthly Active Users", pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("MAU")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        add_headroom(ax)
        save_fig(fig, fig_dir / "monthly_active_users.png")
