#!/usr/bin/env bash

set -euo pipefail

doc_dir="$1"
doc_file="$2"
doc_name="${doc_file%.*}"
build_dir="$doc_dir/build"
tmp_build="$(mktemp -d /tmp/drl-swing-options-build.XXXXXX)"

cleanup() {
	rm -rf "$tmp_build"
}

trap cleanup EXIT

mkdir -p "$build_dir"

# Reuse prior build state so latexmk can stay incremental while writing locally.
find "$build_dir" -maxdepth 1 \( \
	-name "$doc_name.aux" -o \
	-name "$doc_name.bbl" -o \
	-name "$doc_name.blg" -o \
	-name "$doc_name.fdb_latexmk" -o \
	-name "$doc_name.fls" -o \
	-name "$doc_name.log" -o \
	-name "$doc_name.out" -o \
	-name "$doc_name.run.xml" -o \
	-name "$doc_name.synctex.gz" -o \
	-name "$doc_name.toc" \
\) -exec cp {} "$tmp_build/" \;

cd "$doc_dir"
BIBINPUTS="$doc_dir:" BSTINPUTS="$doc_dir:" latexmk \
	-pdf \
	-interaction=nonstopmode \
	-file-line-error \
	-outdir="$tmp_build" \
	"$doc_file"

find "$tmp_build" -maxdepth 1 \( \
	-name "$doc_name.aux" -o \
	-name "$doc_name.bbl" -o \
	-name "$doc_name.blg" -o \
	-name "$doc_name.fdb_latexmk" -o \
	-name "$doc_name.fls" -o \
	-name "$doc_name.log" -o \
	-name "$doc_name.out" -o \
	-name "$doc_name.pdf" -o \
	-name "$doc_name.run.xml" -o \
	-name "$doc_name.synctex.gz" -o \
	-name "$doc_name.toc" \
\) -exec cp {} "$build_dir/" \;