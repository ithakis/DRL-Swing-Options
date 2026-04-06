#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
paper_dir="$repo_dir/Paper"

rm -rf "$paper_dir/build"

find "$paper_dir" -maxdepth 1 \( \
  -name '*.aux' -o \
  -name '*.bbl' -o \
  -name '*.blg' -o \
  -name '*.fdb_latexmk' -o \
  -name '*.fls' -o \
  -name '*.log' -o \
  -name '*.out' -o \
  -name '*.pdf' -o \
  -name '*.synctex.gz' \
\) -delete