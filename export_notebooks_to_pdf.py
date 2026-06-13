#!/usr/bin/env python
"""Convert notebooks to PDF with forced light (white) theme and hidden code cells."""

import subprocess
import sys
from pathlib import Path
import os

NB_DIR = Path(__file__).parent / "Jupyter Notebooks"
notebooks = [
    "3: Validation 3: Semi-Analytical Kernel & RL Pricer.ipynb",
    "Hedging.ipynb",
]

def find_chrome_path():
    possible_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
    ]
    for p in possible_paths:
        if Path(p).exists():
            return p
    return None

# Custom CSS to inject once we purge the custom dark CSS, ensuring absolute white background and crisp dark text.
# IMPORTANT: td/th are deliberately excluded from background-color !important so that pandas Styler
# inline background-gradient colors (set via element style="background-color:...") are not overridden.
LIGHT_THEME_CSS = """
<style>
/* Prevent dark mode preferences from leaking into headless print */
@media print, screen {
  @page {
    size: letter;
    margin: 0.42in 0.34in;
  }

  /* CSS variables: force light theme tokens */
  :root, html, body {
    --jp-layout-color0: #ffffff !important;
    --jp-layout-color1: #ffffff !important;
    --jp-layout-color2: #ffffff !important;
    --jp-layout-color3: #ffffff !important;
    --jp-layout-color4: #ffffff !important;
    --jp-content-font-color0: #000000 !important;
    --jp-content-font-color1: #111111 !important;
    --jp-content-font-color2: #222222 !important;
    --jp-content-font-color3: #333333 !important;
    --jp-mirror-editor-keyword-color: #000077 !important;
    --jp-mirror-editor-string-color: #007700 !important;
    --jp-mirror-editor-variable-color: #000000 !important;
    color-scheme: light !important;
  }

  /* White background for layout containers — NOT td/th so Styler colors survive */
  html, body, div, p, pre, span, a, ul, li, ol {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #000000 !important;
  }

  /* Reset notebook-specific container backgrounds */
  #notebook, .notebook, .jp-Notebook, .jp-Cell, .jp-CellContainer,
  .jp-OutputArea, .jp-OutputArea-child, .jp-OutputPrompt,
  .jp-RenderedHTMLCommon, div.body, body.notebook_app {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #000000 !important;
  }

  /* Headings and prose: force dark text */
  h1, h2, h3, h4, h5, h6, p, pre, li, a {
    color: #000000 !important;
    text-shadow: none !important;
  }

  /* Tables: clean borders and readable text, but let Styler inline background-color win.
     We do NOT set background-color here so pandas background_gradient cells keep their colors. */
  .jp-RenderedHTMLCommon table {
    border-collapse: collapse !important;
    font-size: 9.5px !important;
    line-height: 1.16 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    max-width: 100% !important;
    table-layout: fixed !important;
    width: 100% !important;
  }
  .jp-RenderedHTMLCommon table[id^="T_"] {
    font-size: 8.6px !important;
  }
  .jp-RenderedHTMLCommon table.dataframe {
    font-size: 7.1px !important;
  }
  .jp-RenderedMarkdown table:not(.dataframe) {
    font-size: 10px !important;
  }
  table thead tr {
    background-color: #f1f3f5 !important;
  }
  th {
    color: #000000 !important;
    font-weight: bold !important;
    border: 1px solid #ced4da !important;
    font-size: inherit !important;
    line-height: 1.16 !important;
    overflow-wrap: anywhere !important;
    padding: 3px 5px !important;
    text-align: center !important;
    white-space: normal !important;
  }
  td {
    /* color falls back to the element's inline style when Styler sets it (e.g. white on red) */
    border: 1px solid #dee2e6 !important;
    font-size: inherit !important;
    line-height: 1.16 !important;
    overflow-wrap: anywhere !important;
    padding: 3px 5px !important;
    white-space: normal !important;
  }
  table.dataframe th, table.dataframe td {
    line-height: 1.1 !important;
    padding: 2px 3px !important;
  }
  /* Zebra stripes for plain (unstyled) dataframe tables */
  table.dataframe tbody tr:nth-child(even) {
    background-color: #f8f9fa !important;
  }
  table.dataframe tbody tr:nth-child(odd) {
    background-color: #ffffff !important;
  }
  table.dataframe tbody tr:hover {
    background-color: #e9ecef !important;
  }
  .jp-RenderedHTMLCommon table[id^="T_"] th,
  .jp-RenderedHTMLCommon table[id^="T_"] td {
    padding: 2px 4px !important;
  }

  /* Better proportions for four-column markdown summary tables. */
  .jp-RenderedMarkdown table:has(thead th:nth-child(4):last-child) th:nth-child(1),
  .jp-RenderedMarkdown table:has(thead th:nth-child(4):last-child) td:nth-child(1) {
    width: 16% !important;
  }
  .jp-RenderedMarkdown table:has(thead th:nth-child(4):last-child) th:nth-child(2),
  .jp-RenderedMarkdown table:has(thead th:nth-child(4):last-child) td:nth-child(2) {
    width: 34% !important;
  }
  .jp-RenderedMarkdown table:has(thead th:nth-child(4):last-child) th:nth-child(3),
  .jp-RenderedMarkdown table:has(thead th:nth-child(4):last-child) td:nth-child(3) {
    width: 30% !important;
  }
  .jp-RenderedMarkdown table:has(thead th:nth-child(4):last-child) th:nth-child(4),
  .jp-RenderedMarkdown table:has(thead th:nth-child(4):last-child) td:nth-child(4) {
    width: 20% !important;
  }
  /* Rows with no Styler color: ensure readable dark text */
  tr {
    background-color: inherit;
    color: #111111;
  }

  .jp-RenderedHTMLCommon,
  .jp-OutputArea-output {
    max-width: 100% !important;
    overflow-x: visible !important;
  }

  /* Page breaks: keep tables together where possible */
  table { page-break-inside: auto; }
  tr    { page-break-inside: avoid; page-break-after: auto; }

  /* Remove headers or footers added by custom.css */
  #header, .header-bar, #tree-selector, #maintoolbar, .toolbar_info {
    display: none !important;
  }
}
</style>
"""

chrome_path = find_chrome_path()
if not chrome_path:
    print("✗ Failure: Google Chrome or Microsoft Edge was not found on your system.")
    print("Please install Chrome or Edge to enable headless light-theme PDF prints.")
    sys.exit(1)

for nb in notebooks:
    nb_path = NB_DIR / nb
    if not nb_path.exists():
        print(f"[skip] {nb} not found")
        continue

    print(f"Converting {nb} to light-theme PDF with hidden code cells...")
    
    # 1. Convert notebook to intermediate HTML
    html_path = nb_path.with_suffix(".html")
    html_result = subprocess.run([
        sys.executable, "-m", "nbconvert",
        "--to", "html",
        "--no-input",
        str(nb_path),
        "--output-dir", str(NB_DIR),
    ], capture_output=True, text=True)

    if html_result.returncode != 0:
        print(f"✗ Failed to convert {nb} to intermediate HTML: {html_result.stderr}")
        continue

    # 2. Read HTML and aggressively purge custom dark.css content
    if html_path.exists():
        try:
            content = html_path.read_text(encoding="utf-8")
            
            # Find and purge user's custom.css stylesheet to ensure light theme fallback
            custom_css_path = Path("~/.jupyter/custom/custom.css").expanduser()
            if custom_css_path.exists():
                custom_css_content = custom_css_path.read_text(encoding="utf-8")
                
                # Direct match
                if custom_css_content in content:
                    content = content.replace(custom_css_content, "")
                    print("✓ Successfully purged user's custom dark CSS stylesheet!")
                else:
                    # Match normalized line endings
                    normalized_custom = "\n".join(custom_css_content.splitlines())
                    normalized_content = "\n".join(content.splitlines())
                    if normalized_custom in normalized_content:
                        normalized_content = normalized_content.replace(normalized_custom, "")
                        content = normalized_content
                        print("✓ Successfully purged (normalized) custom dark CSS stylesheet!")
            
            # Inject CSS prior to head closure to force overrides
            if "</head>" in content:
                content = content.replace("</head>", f"{LIGHT_THEME_CSS}</head>")
            else:
                content += LIGHT_THEME_CSS
                
            html_path.write_text(content, encoding="utf-8")
        except Exception as e:
            print(f"⚠ Warning: Could not inject light-theme CSS overrides: {e}")

    # 3. Print HTML to PDF using Chrome's print engine
    pdf_path = nb_path.with_suffix(".pdf")
    tmp_pdf_path = pdf_path.with_name(f"{pdf_path.stem}.tmp.pdf")
    if tmp_pdf_path.exists():
        tmp_pdf_path.unlink()
    chrome_result = subprocess.run([
        chrome_path, "--headless", "--disable-gpu",
        "--color-scheme=light",
        "--virtual-time-budget=10000",  # allow MathJax/KaTeX to finish rendering
        "--print-to-pdf-no-header",     # strip date / title / URL / page-number decorations
        f"--print-to-pdf={tmp_pdf_path}",
        html_path.as_uri()
    ], capture_output=True, text=True)

    # 4. Clean up HTML
    if html_path.exists():
        html_path.unlink()

    # 5. Check if PDF creation succeeded
    if chrome_result.returncode == 0 and tmp_pdf_path.exists() and tmp_pdf_path.stat().st_size > 0:
        tmp_pdf_path.replace(pdf_path)
        print(f"✓ Saved to {pdf_path}")
    else:
        if tmp_pdf_path.exists():
            tmp_pdf_path.unlink()
        print(f"✗ Failed browser PDF generation for {nb}.")
        print(f"  Chrome exit code: {chrome_result.returncode}")
        if chrome_result.stderr:
            print(f"  stderr: {chrome_result.stderr}")

print("\nDone! PDFs are ready to share.")
