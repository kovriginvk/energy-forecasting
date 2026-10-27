# /// script
# requires-python = ">=3.14"
# dependencies = ["mistune==3.2.0", "weasyprint==70.0"]
# ///
"""Render the final Markdown report with inline figures: uv run scripts/2_build_report_pdf.py."""

import re
from html import escape
from pathlib import Path

import mistune
from weasyprint import HTML


def render_figure(match):
    """Keep each Markdown image and its caption together in the PDF."""
    alt, image_path, caption = match.groups()
    return (
        f'<figure><img src="{escape(image_path, quote=True)}" '
        f'alt="{escape(alt, quote=True)}">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def main():
    source = Path(__file__).resolve().parents[1] / "data/final_report/report.md"
    text = re.sub(
        r"!\[([^\]]+)\]\(([^)]+)\)\n\n\*([^\n]+)\*",
        render_figure,
        source.read_text(encoding="utf-8"),
    )
    markdown = mistune.create_markdown(escape=False, plugins=["table"])
    css = """
        @page {
            size: A4; margin: 17mm 18mm;
            @bottom-center {
                content: counter(page);
                font: 9pt "DejaVu Sans"; color: #667085;
            }
        }
        body {
            font-family: "DejaVu Sans", sans-serif;
            font-size: 9pt; line-height: 1.38; color: #182230;
        }
        h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 12pt; }
        h2 { font-size: 12pt; margin: 13pt 0 6pt; }
        h3 { font-size: 10pt; margin: 10pt 0 5pt; }
        h1, h2, h3 { color: #123d50; break-after: avoid; }
        p { margin: 0 0 6pt; orphans: 3; widows: 3; }
        a { color: #1b627d; text-decoration: none; }
        table {
            break-inside: avoid; width: 100%; border-collapse: collapse;
            font-size: 8pt; margin: 7pt 0 9pt;
        }
        th { background: #eaf1f4; color: #123d50; text-align: left; }
        th, td { padding: 4pt 5pt; border-bottom: 0.5pt solid #d6dfe4; }
        tr { break-inside: avoid; }
        code { font-family: "DejaVu Sans Mono"; font-size: 8pt; }
        ol { break-inside: avoid; }
        li { margin-bottom: 6pt; }
        figure { margin: 10pt 0; break-inside: avoid; }
        figure img { width: 100%; display: block; }
        figcaption { margin-top: 6pt; font-size: 8pt; color: #475467; }
    """
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Forecasting Electricity Carbon Intensity in the Netherlands</title>"
        f"<style>{css}</style></head><body>{markdown(text)}</body></html>"
    )
    output = source.with_suffix(".pdf")
    HTML(string=html, base_url=str(source.parent)).write_pdf(output)
    print(f"Created {output}")


if __name__ == "__main__":
    main()
