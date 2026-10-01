"""Build artifacts/final/report_li/index.html from artifacts/final/base/li/li.json (+ figures)."""
import html
import json
import os
import re
import shutil
import sys

ROOT = "/home/rrahman/BT/Tabula Rasa/artifacts/final"
OUT = os.path.join(ROOT, "report_li")
ENV = "base"
S = json.load(open(os.path.join(ROOT, ENV, "li", "li.json")))
BODY = open(sys.argv[1]).read()           # the hand-written prose, with {{TABLE:...}} / {{V:...}} slots

os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
for f in ("fig_li_vs_jsr.png", "fig_li_damage_vs_dr.png", "fig_iq_per50.png", "fig_iq_per90.png"):
    shutil.copy(os.path.join(ROOT, ENV, "li", f), os.path.join(OUT, "figs", f"{ENV}_{f}"))

prev = open(os.path.join(ROOT, "report", "index.html")).read()
style = prev[prev.index("<link rel=\"preconnect\""):prev.index("</style>")]
style += """
.k-li { background: var(--k-control); } .k-grey { background: var(--k-grey); }
.k-dash { background: repeating-linear-gradient(90deg, var(--k-grey) 0 6px, transparent 6px 9px); }
.k-cnn-d { background: repeating-linear-gradient(90deg, var(--k-cnn) 0 6px, transparent 6px 9px); }
.k-li-d { background: repeating-linear-gradient(90deg, var(--k-control) 0 6px, transparent 6px 9px); }
table.res td.lbl { white-space: nowrap; }
details { margin: 8px 0 18px; } summary { cursor: pointer; font-family: var(--font-display); font-weight: 600; color: var(--fg2); }
"""
KEY = {"omniscient_e1": "k-genie", "tone": "k-grey", "pulse_comb": "k-dash", "noise": "k-noise",
       "pulsed_p0.25": "k-amuru", "cnn_li": "k-li-d", "cnn_b10": "k-cnn-d", "cnn_li_grey": "k-li",
       "cnn_b10_grey": "k-cnn"}
LEVELS = ["0.1", "0.5", "0.9"]


def mean_sd(v, pct=False, nd=3):
    v = [x for x in v if x is not None and x == x]
    if not v:
        return None, None
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else None
    return m, sd


def cell(v, pct=False, nd=3, shade=False):
    m, sd = mean_sd(v)
    if m is None:
        return '<td class="na">—</td>'
    s = 100 if pct else 1
    nd = 1 if pct else nd
    txt = f"{s * m:.{nd}f}" + (f'<span class="pm">± {s * sd:.{nd}f}</span>' if sd is not None else "")
    st = f' style="--v:{100 * m:.1f}"' if shade else ""
    return f'<td class="num"{st}>{txt}</td>'


def label(key):
    e = S["attackers"][key]
    return f'<td class="lbl"><span class="key {KEY[key]}"></span>{html.escape(e["label"])}</td>'


def table_a(k):
    rows = []
    for key, e in S["attackers"].items():
        a = e[k]["window_a"]
        rows.append("<tr>" + label(key) + cell(a["recall"], shade=True) + cell(a["mean_excess_per"], nd=2)
                    + cell([m["dr"] for m in a["li"]], pct=True) + cell([m["precision"] for m in a["li"]])
                    + cell([m["f_score"] for m in a["li"]]) + "</tr>")
    p = S["pooled_li_four"][k]
    rows.append(f'<tr><td class="lbl"><strong>Li\'s four pooled</strong> (Li\'s two-class set, 762 : 816)</td>'
                f'<td class="num">{p["recall"]:.3f}</td><td class="na">—</td><td class="num">{100 * p["dr"]:.1f}</td>'
                f'<td class="num">{p["precision"]:.3f}</td><td class="num">{p["f_score"]:.3f}</td></tr>')
    head = ('<tr><th>jammer</th><th class="num">recall</th><th class="num">mean excess PER</th>'
            '<th class="num">DR (eq. 2a) %</th><th class="num">precision</th><th class="num">F-score</th></tr>')
    return f'<div class="tablewrap"><table class="res"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def table_b(k):
    rows = []
    for key, e in S["attackers"].items():
        r = "<tr>" + label(key)
        for l in LEVELS:
            m = e[k]["matched"][l]
            r += cell(m["jsr"], nd=1) + cell(m["recall"], shade=True)
        li = [m for m in e[k]["matched"]["0.9"]["li"] if m]
        r += cell([m["dr"] for m in li], pct=True) + cell([m["f_score"] for m in li]) + "</tr>"
        rows.append(r)
    head = ('<tr><th rowspan="2">jammer</th><th colspan="2" class="num">PER 0.1</th><th colspan="2" class="num">PER 0.5</th>'
            '<th colspan="4" class="num">PER 0.9 = Li\'s damage</th></tr><tr>'
            + '<th class="num">JSR dB</th><th class="num">recall</th>' * 3
            + '<th class="num">DR %</th><th class="num">F-score</th></tr>')
    return f'<div class="tablewrap"><table class="res"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def value(path):
    """{{V:attacker/decision/matched/0.9/recall}} -> mean (± sd) as plain text."""
    parts = path.split("/")
    node = S["attackers"] if parts[0] not in ("far", "pooled_li_four") else S
    for p in parts:
        node = node[p]
    if isinstance(node, list):
        m, sd = mean_sd(node)
        return f"{m:.3f}" + (f" ± {sd:.3f}" if sd is not None else "")
    return f"{node:.4g}" if isinstance(node, float) else str(node)


body = re.sub(r"\{\{TABLE:(A|B):([^}]+)\}\}", lambda m: (table_a if m[1] == "A" else table_b)(m[2]), BODY)
body = re.sub(r"\{\{V:([^}]+)\}\}", lambda m: value(m[1]), body)
page = "<title>Li-Protocol CNN Readout</title>\n" + style + "</style>\n" + body
open(os.path.join(OUT, "index.html"), "w").write(page)
left = re.findall(r"\{\{[^}]+\}\}", page)
print(f"wrote {OUT}/index.html ({len(page)} bytes); unfilled slots: {left}")
