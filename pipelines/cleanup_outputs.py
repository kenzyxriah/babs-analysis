from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


IMG_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>.*?)\)")

DEFAULT_KEEP_TABLES: set[str] = {
    # Psychographic / inquiries
    "inquiry_volume_by_month.csv",
    "inquiry_intent_tags.csv",
    "inquiry_conversion.csv",
    "skill_gap_extractions.csv",
    # Financial
    "payments_status_by_month.csv",
    "revenue_by_month.csv",
    "revenue_waterfall.csv",
    "discount_hook_summary.csv",
    "payment_plan_engagement.csv",
    "paid_in_full_by_product.csv",
    "paid_revenue_by_product.csv",
    "product_revenue_pareto.csv",
    "custom_product_revenue_by_month.csv",
    "payments_received_by_month.csv",
    # Behavioral
    "enrollments_by_course_month.csv",
    "enrollments_by_product_month.csv",
    "course_completion_summary.csv",
    "completion_threshold_breakdown.csv",
    "assignment_completion_by_course.csv",
    "assignment_submission_summary.csv",
    "time_to_submit_distribution.csv",
    "engagement_trends_over_time.csv",
    "session_attendance_summary.csv",
    "login_monthly_active.csv",
    "session_join_rate_trends.csv",
    "career_goal_buckets.csv",
    "product_adoption_summary.csv",
    "program_selection_summary.csv",
    "tag_category_coverage.csv",
    "specialization_tag_revenue.csv",
    # Product / ops
    "ops_gap_report.csv",
    "exception_duration_summary.csv",
    "sales_lag_distribution.csv",
    "agreement_compliance_distribution.csv",
    "agreement_compliance_time.csv",
}


@dataclass(frozen=True)
class CleanupResult:
    deleted_figures: int
    deleted_tables: int
    kept_figures: int
    kept_tables: int


def _paths_referenced_in_report(report_md: str) -> set[str]:
    refs: set[str] = set()
    for m in IMG_RE.finditer(report_md):
        rel = (m.group("path") or "").strip()
        if rel:
            refs.add(rel.replace("\\", "/"))
    # Also support raw paths in markdown (rare, but useful for tables if we link them later)
    for m in re.finditer(r"(output/(?:figures|tables)/[-A-Za-z0-9_ .\\\\/]+)", report_md):
        refs.add(m.group(1).strip().replace("\\", "/"))
    return refs


def cleanup_outputs(
    *,
    base_dir: Path,
    report_md_path: Path,
    keep_tables: set[str] | None = None,
) -> CleanupResult:
    """
    Deletes redundant files under analytics/output/** so only executive deliverables remain.

    Rules:
    - Figures: keep only images referenced by report.md.
    - Tables: keep only an allowlist (plus any tables referenced by report.md).
    """
    report_md = report_md_path.read_text(encoding="utf-8") if report_md_path.exists() else ""
    refs = _paths_referenced_in_report(report_md)

    fig_dir = base_dir / "output" / "figures"
    table_dir = base_dir / "output" / "tables"

    keep_figs: set[str] = set()
    keep_set = keep_tables or DEFAULT_KEEP_TABLES
    keep_tabs: set[str] = {f"output/tables/{name}" for name in keep_set}

    for rel in refs:
        norm = rel.replace("\\", "/")
        if norm.startswith("output/figures/"):
            keep_figs.add(norm)
        if norm.startswith("output/tables/"):
            keep_tabs.add(norm)

    deleted_figures = 0
    deleted_tables = 0

    # Figures
    if fig_dir.exists():
        for p in fig_dir.glob("*.png"):
            rel = f"output/figures/{p.name}"
            if rel not in keep_figs:
                try:
                    p.unlink()
                    deleted_figures += 1
                except Exception:
                    pass

    # Tables
    if table_dir.exists():
        for p in table_dir.glob("*.csv"):
            rel = f"output/tables/{p.name}"
            if rel not in keep_tabs:
                try:
                    p.unlink()
                    deleted_tables += 1
                except Exception:
                    pass

    kept_figures = len(list(fig_dir.glob("*.png"))) if fig_dir.exists() else 0
    kept_tables = len(list(table_dir.glob("*.csv"))) if table_dir.exists() else 0

    return CleanupResult(
        deleted_figures=deleted_figures,
        deleted_tables=deleted_tables,
        kept_figures=kept_figures,
        kept_tables=kept_tables,
    )
