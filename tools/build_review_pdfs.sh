#!/usr/bin/env bash
# Build the three review PDFs from the single manuscript source, toggling the
# flag-guarded review-markup machinery (see the preamble of DRL_Swing_Options.tex):
#   DRL_Swing_Options.pdf                 -- clean (no flags; == the normal build)
#   DRL_Swing_Options_highlighted.pdf     -- light-yellow highlights of changes vs pre-DP
#   DRL_Swing_Options_highlighted_notes.pdf -- highlights + red [CHG: ...] notes
set -euo pipefail

doc_dir="$(cd "$(dirname "$0")/../Paper" && pwd)"
doc="DRL_Swing_Options"
build_dir="$doc_dir/build"
mkdir -p "$build_dir"
cd "$doc_dir"

build() {   # $1 = jobname, $2 = TeX \def prefix
  local job="$1" defs="$2"
  local tmp; tmp="$(mktemp -d /tmp/drl-review-build.XXXXXX)"
  BIBINPUTS="$doc_dir:" BSTINPUTS="$doc_dir:" latexmk -f -pdf \
    -interaction=nonstopmode -file-line-error -outdir="$tmp" -jobname="$job" \
    -pdflatex="pdflatex %O \"${defs}\\input{%S}\"" "$doc.tex" >/dev/null 2>&1 || true
  if [[ -f "$tmp/$job.pdf" ]]; then
    cp "$tmp/$job.pdf" "$build_dir/"
    echo "  built $build_dir/$job.pdf ($(grep -c . < /dev/null || true)$(pdfinfo "$tmp/$job.pdf" 2>/dev/null | awk '/Pages/{print $2" pages"}'))"
  else
    echo "  FAILED $job (see $tmp/$job.log)"; return 1
  fi
  rm -rf "$tmp"
}

echo "Building review PDFs..."
build "$doc"                    ""
build "${doc}_highlighted"       "\\def\\HLon{}"
build "${doc}_highlighted_notes" "\\def\\HLon{}\\def\\NOTEon{}"
echo "Done."
