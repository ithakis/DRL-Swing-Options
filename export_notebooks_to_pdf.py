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

# Custom CSS to inject once we purge the custom dark CSS, ensuring absolute white background and crisp dark text
LIGHT_THEME_CSS = """
<style>
/* Prevent dark mode preferences from leaking into headless print */
@media print, screen {
  :root, html, body, div, p, pre, span, a, h1, h2, h3, h4, h5, h6, table, tr, td, th, ul, li, ol {
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
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #000000 !important;
  }
  
  /* Reset container backgrounds specifically to be white */
  #notebook, .notebook, .jp-Notebook, .jp-Cell, .jp-CellContainer, .jp-OutputArea, .jp-OutputArea-child, .jp-OutputPrompt, .jp-RenderedHTMLCommon, div.body, body.notebook_app {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #000000 !important;
  }
  
  /* Guarantee readable text headings */
  h1, h2, h3, h4, h5, h6, p, pre, td, th, li, a, span {
    color: #000000 !important;
    text-shadow: none !important;
  }

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
    chrome_result = subprocess.run([
        chrome_path, "--headless", "--disable-gpu",
        "--color-scheme=light",  # browser level light theme constraint
        f"--print-to-pdf={pdf_path}",
        str(html_path)
    ], capture_output=True, text=True)

    # 4. Clean up HTML
    if html_path.exists():
        html_path.unlink()

    # 5. Check if PDF creation succeeded
    if pdf_path.exists():
        print(f"✓ Saved to {pdf_path}")
    else:
        print(f"✗ Failed browser PDF generation for {nb}. Error: {chrome_result.stderr}")

print("\nDone! PDFs are ready to share.")
