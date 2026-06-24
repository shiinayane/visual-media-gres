#!/usr/bin/env bash
# Build report/report.pdf from report/report.md (pandoc -> HTML -> weasyprint).
# Run from anywhere; paths inside report.md are relative to report/.
set -euo pipefail
cd "$(dirname "$0")/../report"

CSS=$(cat <<'EOF'
@page { size: A4; margin: 1.8cm 1.9cm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10.3pt; line-height: 1.34; color:#111; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom:4px; }
h2 { font-size: 12.5pt; margin-top: 1.1em; color:#1a1a1a; border-bottom:1px solid #bbb; padding-bottom:2px;}
h3 { font-size: 11pt; color:#333; margin-bottom:.2em;}
code { background:#f2f2f2; padding:0 2px; font-size:9pt; }
pre { background:#f6f6f6; padding:6px; font-size:8.6pt; overflow:hidden; }
table { border-collapse: collapse; font-size: 9.4pt; margin:.5em 0; }
th,td { border:1px solid #999; padding:2px 7px; text-align:center; }
th { background:#eee; }
img { display:inline-block; margin:6px 2px; }
blockquote { color:#444; border-left:3px solid #ccc; padding-left:8px; font-size:9.5pt; }
EOF
)
echo "$CSS" > /tmp/report.css

pandoc report.md \
  --from gfm --to html5 --standalone \
  --metadata title="" \
  --css /tmp/report.css \
  --pdf-engine=weasyprint \
  -o report.pdf
echo "[built] report/report.pdf"
