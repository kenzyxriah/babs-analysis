from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, portrait
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT


IMG_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>.*?)\)")
MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
MD_OL_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
PAGEBREAK_MARKERS = {"---PAGEBREAK---", "<!--PAGEBREAK-->", "<!-- PAGEBREAK -->"}


@dataclass
class Block:
    kind: str
    data: object


def _register_fonts() -> tuple[str, str]:
    cambria_ttc = Path(r"C:\Windows\Fonts\cambria.ttc")
    cambria_bold = Path(r"C:\Windows\Fonts\cambriab.ttf")

    body_font = "Times-Roman"
    body_bold = "Times-Bold"

    try:
        if cambria_ttc.exists():
            pdfmetrics.registerFont(TTFont("Cambria", str(cambria_ttc), subfontIndex=0))
            body_font = "Cambria"
        if cambria_bold.exists():
            pdfmetrics.registerFont(TTFont("Cambria-Bold", str(cambria_bold)))
            body_bold = "Cambria-Bold"
    except Exception:
        # If font registration fails, fall back to built-in fonts.
        pass

    return body_font, body_bold


def _parse_markdown(md: str) -> list[Block]:
    lines = md.splitlines()
    blocks: list[Block] = []

    i = 0
    para_buf: list[str] = []

    def flush_para() -> None:
        nonlocal para_buf
        text = "\n".join([l.strip() for l in para_buf]).strip()
        if text:
            blocks.append(Block("paragraph", text))
        para_buf = []

    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Explicit page breaks (for 1-page exec summary, etc.)
        if line.strip() in PAGEBREAK_MARKERS:
            flush_para()
            blocks.append(Block("pagebreak", None))
            i += 1
            continue

        # Headings
        if line.startswith("#"):
            flush_para()
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            blocks.append(Block("heading", (level, title)))
            i += 1
            continue

        # Images
        m = IMG_RE.search(line)
        if m:
            flush_para()
            blocks.append(Block("image", (m.group("alt").strip(), m.group("path").strip())))
            i += 1
            continue

        # Tables
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            flush_para()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            # Remove separator rows like | --- | --- |
            rows = []
            for tl in table_lines:
                parts = [p.strip() for p in tl.strip("|").split("|")]
                if all(p.replace("-", "").strip() == "" for p in parts):
                    continue
                rows.append(parts)

            if rows:
                blocks.append(Block("table", rows))
            continue

        # Blank line = paragraph boundary
        if not line.strip():
            flush_para()
            i += 1
            continue

        # Bullets: keep as paragraph text; ReportLab will wrap
        para_buf.append(line)
        i += 1

    flush_para()
    return blocks


def export_pdf(
    report_md_path: Path,
    pdf_path: Path,
    base_dir: Path,
) -> None:
    body_font, body_bold = _register_fonts()

    md = report_md_path.read_text(encoding="utf-8")
    blocks = _parse_markdown(md)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName=body_bold,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName=body_bold,
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName=body_bold,
        fontSize=12.5,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=14,
        bulletIndent=6,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=10,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#444444"),
        spaceAfter=10,
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=portrait(letter),
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title="Babskenky and Company Report",
        author="Babskenky",
    )

    story = []
    fig_no = 0
    tbl_no = 0

    current_section = None

    table_cell_style = ParagraphStyle(
        "TableCell",
        fontName=body_font,
        fontSize=10.5,
        leading=12,
        alignment=TA_LEFT,
    )

    for block in blocks:
        if block.kind == "heading":
            level, title = block.data  # type: ignore[misc]
            if level == 1:
                story.append(Paragraph(title, title_style))
            elif level == 2:
                story.append(Paragraph(title, h1_style))
            else:
                story.append(Paragraph(title, h2_style))
            current_section = title
            continue

        if block.kind == "pagebreak":
            story.append(PageBreak())
            continue

        if block.kind == "paragraph":
            text: str = block.data  # type: ignore[assignment]
            text = MD_BOLD_RE.sub(r"<b>\1</b>", text)
            # Convert markdown bullets to simple bullets
            lines = text.splitlines()
            if all(l.strip().startswith("-") for l in lines if l.strip()):
                for l in lines:
                    if not l.strip():
                        continue
                    content = l.strip().lstrip("-").strip()
                    story.append(Paragraph(f"• {content}", bullet_style))
                story.append(Spacer(1, 6))
            elif all(MD_OL_RE.match(l.strip()) for l in lines if l.strip()):
                for l in lines:
                    if not l.strip():
                        continue
                    m = MD_OL_RE.match(l.strip())
                    if not m:
                        continue
                    n, content = m.group(1), m.group(2)
                    story.append(Paragraph(f"{n}. {content}", bullet_style))
                story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(text.replace("\n", " "), body_style))
            continue

        if block.kind == "table":
            rows: list[list[str]] = block.data  # type: ignore[assignment]
            headers = rows[0]
            # Wrap cells so long content doesn't blow out page width
            data: list[list[Paragraph]] = []
            data.append([Paragraph(MD_BOLD_RE.sub(r"<b>\1</b>", h), table_cell_style) for h in headers])
            for row in rows[1:]:
                data.append([Paragraph(MD_BOLD_RE.sub(r"<b>\1</b>", str(c)), table_cell_style) for c in row])

            tbl_no += 1
            caption = f"Table {tbl_no}: {current_section or 'Summary'}"
            story.append(Paragraph(caption, caption_style))

            # Create table with wrapping; allocate width proportional to content length.
            col_count = max(len(headers), 1)
            col_max = [0] * col_count
            for r_i, row in enumerate(rows):
                for c_i in range(min(len(row), col_count)):
                    col_max[c_i] = max(col_max[c_i], len(str(row[c_i] or "")))
            weights = [max(m, 8) for m in col_max]
            total_w = float(sum(weights)) or 1.0
            col_widths = [(w / total_w) * doc.width for w in weights]
            t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=col_widths)
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), body_bold),
                        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#aaaaaa")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            # Zebra striping for readability
            try:
                for rr in range(1, len(rows)):
                    if rr % 2 == 0:
                        t.setStyle(TableStyle([("BACKGROUND", (0, rr), (-1, rr), colors.HexColor("#f4f6f8"))]))
            except Exception:
                pass
            story.append(t)
            story.append(Spacer(1, 10))
            continue

        if block.kind == "image":
            alt, rel = block.data  # type: ignore[misc]
            img_path = (base_dir / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            if not img_path.exists():
                # Skip missing images but leave a note
                story.append(Paragraph(f"[Missing image: {rel}]", body_style))
                continue

            fig_no += 1
            max_w = doc.width
            img = Image(str(img_path))
            # Scale proportionally
            iw, ih = img.imageWidth, img.imageHeight
            scale = min(max_w / iw, 1.0)
            img.drawWidth = iw * scale
            img.drawHeight = ih * scale

            story.append(img)
            story.append(Paragraph(f"Figure {fig_no}: {alt}", caption_style))
            continue

    doc.build(story)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    export_pdf(
        report_md_path=base / "report.md",
        pdf_path=base / "Babskenky and Company Report.pdf",
        base_dir=base,
    )
