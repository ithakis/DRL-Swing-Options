---
name: notebook-pdf-export
description: "Use when converting Jupyter Notebooks (.ipynb) to PDFs with all code cells hidden (collapsed), ensuring a clean, publication-grade light (white) theme regardless of the user's background or system preferences."
---

# Notebook PDF Export (White-Theme, Code-Hidden)

Use this skill when a user asks to export any Jupyter Notebook in the repository to a PDF for sharing where code implementation details should be hidden and only Markdown cells, math equations, tables, and figures are visible.

## Key Goals & Properties
1. **No Code Cells (`--no-input`):** Keep code visual noise completely hidden, focusing entirely on markdown narration, LaTeX equations, and generated code figures.
2. **Forced Light Theme:** Output PDFs must render in a clean, professional print-ready light theme (white background, dark text/math/plots) even if the user's IDE, browser, or system is currently running in a Dark Theme.
3. **Robust Browser fallback:** Avoid brittle local LaTeX compiler issues (such as missing `tcolorbox.sty` errors) by using modern `nbconvert` HTML output coupled with a headless browser (`Google Chrome` or `Microsoft Edge`) PDF engine.

## Action Steps

### Standard Repo Invocation
The codebase includes a unified CLI script `export_notebooks_to_pdf.py` at the root which converts the designated notebook(s) to professional Light-Theme PDFs. Always run this script using the active Python environment:

```bash
# General run via the active workspace environment
python export_notebooks_to_pdf.py
```

### Script Internals & Custom Flow
If you need to manually print or customize any other notebook to a PDF with hidden code cells and forced light theme, execute the following three mechanical steps:

1. **Convert Jupyter to intermediate HTML with no-input cells:**
   ```bash
   jupyter nbconvert --to html --no-input "path/to/notebook.ipynb" --output-dir "path/to/output_dir"
   ```

2. **Inject CSS override block** right before the closing `</head>` tag of the generated HTML file to force a white background and light-theme text colors across `@media (prefers-color-scheme: dark)`:
   ```html
   <style>
   :root {
     --jp-layout-color0: #ffffff !important;
     --jp-layout-color1: #ffffff !important;
     --jp-layout-color2: #f5f5f5 !important;
     --jp-layout-color3: #e0e0e0 !important;
     --jp-layout-color4: #bdbdbd !important;
     --jp-content-font-color0: rgba(0, 0, 0, 0.88) !important;
     --jp-content-font-color1: rgba(0, 0, 0, 0.8) !important;
     --jp-content-font-color2: rgba(0, 0, 0, 0.54) !important;
     --jp-content-font-color3: rgba(0, 0, 0, 0.38) !important;
     color-scheme: light !important;
   }
   body, .jp-Notebook, .jp-Cell, .jp-CellContainer, .jp-OutputArea, .jp-OutputArea-child, .jp-OutputPrompt, .jp-RenderedHTMLCommon {
     background-color: #ffffff !important;
     background: #ffffff !important;
     color: #000000 !important;
   }
   div.text_cell_render, p, h1, h2, h3, h4, h5, h6, span, td, th, li, a, pre {
     color: #000000 !important;
   }
   @media (prefers-color-scheme: dark) {
     :root {
       --jp-layout-color0: #ffffff !important;
       --jp-layout-color1: #ffffff !important;
       --jp-layout-color2: #f5f5f5 !important;
       --jp-layout-color3: #e0e0e0 !important;
       --jp-layout-color4: #bdbdbd !important;
       --jp-content-font-color0: rgba(0, 0, 0, 0.88) !important;
       --jp-content-font-color1: rgba(0, 0, 0, 0.8) !important;
       --jp-content-font-color2: rgba(0, 0, 0, 0.54) !important;
       --jp-content-font-color3: rgba(0, 0, 0, 0.38) !important;
       color-scheme: light !important;
     }
     body, .jp-Notebook, .jp-Cell, .jp-CellContainer, .jp-OutputArea, .jp-OutputArea-child, .jp-OutputPrompt, .jp-RenderedHTMLCommon {
       background-color: #ffffff !important;
       background: #ffffff !important;
       color: #000000 !important;
     }
     div.text_cell_render, p, h1, h2, h3, h4, h5, h6, span, td, th, li, a, pre {
       color: #000000 !important;
     }
   }
   </style>
   ```

3. **Convert modern HTML to PDF using a headless chromium browser print engine:**
   ```bash
   # On macOS
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --color-scheme=light --print-to-pdf="path/to/output.pdf" "path/to/notebook.html"
   ```

4. **Clean up:** Delete the intermediate `.html` file after the PDF has been successfully written on the disk.

## Practical Guardrails

- Never instruct user to type long commands. Execute `export_notebooks_to_pdf.py` as a single terminal task.
- Ensure the output PDF files are linked for easy one-click download/navigation by the user.
