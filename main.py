from __future__ import annotations

import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.config.settings import get_settings
from analytics.io.loaders import load_all
from analytics.models.schema import Context
from analytics.pipelines.build_tables import build_tables
from analytics.pipelines.build_figures import build_figures
from analytics.pipelines.build_report import build_report
from analytics.pipelines.export_pdf import export_pdf
from analytics.pipelines.cleanup_outputs import cleanup_outputs


async def main() -> None:
    settings = get_settings()
    data = load_all(settings.data_dir)
    ctx = Context(settings=settings, data=data)

    await build_tables(ctx)
    build_figures(ctx)
    build_report(ctx)
    export_pdf(
        report_md_path=settings.base_dir / "report.md",
        pdf_path=settings.base_dir / "Babskenky and Company Feb 5th 2026 Report.pdf",
        base_dir=settings.base_dir,
    )
    cleanup_outputs(base_dir=settings.base_dir, report_md_path=settings.base_dir / "report.md")

    print("Analytics pipeline completed.")


if __name__ == "__main__":
    asyncio.run(main())
