#!/bin/bash
# Compile the Overleaf clone's paper.tex and report/report.tex into paper_drafts/preview/{paper,report}.pdf, so the
# user can keep both PDFs open and read every change (user, 2026-11-26). Aux files go to net_scratch, never into the
# clone. Run by the Claude Code Stop hook (.claude/settings.json) when a .tex/.bib/.cls in overleaf/ changed, or by hand:
#   paper_drafts/build_preview.sh [--force]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OL="$ROOT/overleaf"
OUT="$ROOT/paper_drafts/preview"
AUX="/itet-stor/rrahman/net_scratch/preview_aux"
mkdir -p "$OUT" "$AUX/paper" "$AUX/report"
[ -d "$OL" ] || exit 0
# Rebuild only when a source is newer than the last preview (the hook fires at the end of every turn).
if [ "${1:-}" != "--force" ] && [ -f "$OUT/paper.pdf" ] && \
   [ -z "$(find "$OL" -path "$OL/.git" -prune -o \( -name '*.tex' -o -name '*.bib' -o -name '*.cls' -o -name '*.pdf' -o -name '*.png' -o -name '*.csv' \) -newer "$OUT/paper.pdf" -print -quit)" ]; then
  exit 0
fi
status=0
( cd "$OL" && LATEXMK_USE_OUTDIR=1 BIBINPUTS=.: timeout 240 latexmk -pdf -outdir="$AUX/paper" -interaction=nonstopmode paper.tex >"$AUX/paper/build.log" 2>&1 )
[ -f "$AUX/paper/paper.pdf" ] && cp "$AUX/paper/paper.pdf" "$OUT/paper.pdf"
perr=$(grep -c '^!' "$AUX/paper/paper.log" 2>/dev/null); perr=${perr:-1}
( cd "$OL/report" && TEXINPUTS=.:..: BIBINPUTS=..: timeout 240 latexmk -pdf -outdir="$AUX/report" -interaction=nonstopmode report.tex >"$AUX/report/build.log" 2>&1 )
[ -f "$AUX/report/report.pdf" ] && cp "$AUX/report/report.pdf" "$OUT/report.pdf"
rerr=$(grep -c '^!' "$AUX/report/report.log" 2>/dev/null); rerr=${rerr:-1}
touch "$OUT/paper.pdf" 2>/dev/null
msg="preview: paper.pdf ($perr LaTeX errors), report.pdf ($rerr LaTeX errors) -> paper_drafts/preview/"
if [ "$perr" != "0" ] || [ "$rerr" != "0" ]; then msg="$msg; logs in $AUX/{paper,report}/"; fi
# A Stop hook shows "systemMessage" to the user; plain stdout when run by hand.
if [ -t 1 ]; then echo "$msg"; else printf '{"systemMessage": "%s"}\n' "$msg"; fi
exit $status
