#!/usr/bin/env python3
"""Render the benchmark comparison chart from an audit results JSON.

Emits two SVGs (light and dark theme) for the README, referenced through a
<picture> element so GitHub serves the right one. The form is a dot plot on
a log axis: the measured ranges span four orders of magnitude, which bar
length cannot encode honestly (bars need a zero baseline; a log axis has
none). bytepair carries the accent hue; the comparison implementations are
deliberately gray. Identity is carried by the row labels, so color never
carries meaning alone.

    chart.py <results.json> <out-light.svg> <out-dark.svg>
"""
import json
import math
import sys

TOOLS = [("bytepair", "bytepair"),
         ("gigatoken", "GigaToken"),
         ("hf-tokenizers", "HF tokenizers"),
         ("bpe-qwen", "bpe-qwen")]

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "accent": "#2a78d6",
              "gray_dot": "#898781"},
    "dark":  {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
              "muted": "#898781", "grid": "#2c2c2a", "accent": "#3987e5",
              "gray_dot": "#898781"},
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

W, GUTTER, RIGHT = 880, 128, 40
ROW_H, TITLE_H, AXIS_H, GAP = 24, 22, 20, 14

def fmt(v):
    if v >= 100: return f"{v:,.0f}"
    if v >= 10: return f"{v:.1f}"
    return f"{v:.2f}"

def build_panels(run):
    c = run["corpora"]
    en = next(k for k in c if k.startswith("enwik8"))
    cjk = next((k for k in c if "cjk" in k), None)

    def metric(corpus, key):
        out = {}
        for tool, _ in TOOLS:
            r = c[corpus]["tools"].get(tool, {})
            out[tool] = r.get(key) if "error" not in r else None
        return out

    panels = [
        ("vocabulary open (ms, log scale)", metric(en, "load_ms"), None),
        (f"single thread, {en} (MB/s, log scale)", metric(en, "st_mbs"), None),
        (f"all {run['threads']} threads, {en} (MB/s, log scale)",
         metric(en, "mt_mbs"), "no batch API"),
    ]
    if cjk:
        panels.append((f"single thread, {cjk}, cache-hostile (MB/s, log scale)",
                       metric(cjk, "st_mbs"), None))
    return panels

def svg(run, theme):
    t = THEMES[theme]
    panels = build_panels(run)
    n_rows = len(TOOLS)
    panel_h = TITLE_H + n_rows * ROW_H + AXIS_H + GAP
    header_h = 64
    H = header_h + panel_h * len(panels)
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{H}" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Benchmark comparison">')
    o.append(f'<rect width="{W}" height="{H}" fill="{t["surface"]}" rx="6"/>')

    cpu = run["cpu"].replace("Intel(R) Xeon(R) CPU ", "Intel Xeon ")
    o.append(f'<text x="20" y="26" font-family=\'{FONT}\' font-size="14" '
             f'font-weight="600" fill="{t["ink"]}">bytepair benchmark</text>')
    o.append(f'<text x="20" y="44" font-family=\'{FONT}\' font-size="11" '
             f'fill="{t["ink2"]}">{cpu}, {run["timestamp"][:10]}. '
             f'Reproduce with: sh bench/audit.sh</text>')
    # legend
    lx = W - 300
    o.append(f'<circle cx="{lx}" cy="22" r="5" fill="{t["accent"]}"/>')
    o.append(f'<text x="{lx+10}" y="26" font-family=\'{FONT}\' '
             f'font-size="10.5" fill="{t["ink2"]}">bytepair</text>')
    o.append(f'<circle cx="{lx+76}" cy="22" r="5" fill="{t["gray_dot"]}"/>')
    o.append(f'<text x="{lx+86}" y="26" font-family=\'{FONT}\' '
             f'font-size="10.5" fill="{t["ink2"]}">comparison implementations'
             f'</text>')

    y0 = header_h
    for title, vals, absent_note in panels:
        o.append(f'<text x="20" y="{y0+13}" font-family=\'{FONT}\' '
                 f'font-size="11" font-weight="600" fill="{t["ink2"]}">'
                 f'{title}</text>')
        plot_x0, plot_x1 = GUTTER, W - RIGHT
        present = [v for v in vals.values() if v]
        lo = 10 ** math.floor(math.log10(min(present)))
        hi = 10 ** math.ceil(math.log10(max(present)))
        span = math.log10(hi) - math.log10(lo)

        def X(v):
            return plot_x0 + (math.log10(v) - math.log10(lo)) / span * \
                   (plot_x1 - plot_x0)

        top, bot = y0 + TITLE_H, y0 + TITLE_H + n_rows * ROW_H
        d = lo
        while d <= hi:
            x = X(d)
            o.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                     f'y2="{bot}" stroke="{t["grid"]}" stroke-width="1"/>')
            label = f"{d:g}"
            o.append(f'<text x="{x:.1f}" y="{bot+14}" font-family=\'{FONT}\' '
                     f'font-size="10" fill="{t["muted"]}" '
                     f'text-anchor="middle" style="font-variant-numeric:'
                     f'tabular-nums">{label}</text>')
            d *= 10
        for k, (tool, name) in enumerate(TOOLS):
            cy = top + k * ROW_H + ROW_H // 2
            weight = "600" if tool == "bytepair" else "400"
            o.append(f'<text x="{GUTTER-10}" y="{cy+4}" '
                     f'font-family=\'{FONT}\' font-size="11" '
                     f'font-weight="{weight}" fill="{t["ink2"]}" '
                     f'text-anchor="end">{name}</text>')
            v = vals.get(tool)
            if not v:
                note = absent_note or "not measured"
                o.append(f'<text x="{plot_x0+6}" y="{cy+4}" '
                         f'font-family=\'{FONT}\' font-size="10" '
                         f'fill="{t["muted"]}" font-style="italic">{note}'
                         f'</text>')
                continue
            x = X(v)
            color = t["accent"] if tool == "bytepair" else t["gray_dot"]
            # a neutral row guide, never a colored stem: on a log axis a
            # stem's length would encode a ratio the eye reads as linear
            o.append(f'<line x1="{plot_x0}" y1="{cy}" x2="{plot_x1}" '
                     f'y2="{cy}" stroke="{t["grid"]}" stroke-width="1"/>')
            o.append(f'<circle cx="{x:.1f}" cy="{cy}" r="5" '
                     f'fill="{color}"/>')
            lw = "600" if tool == "bytepair" else "400"
            fill = t["ink"] if tool == "bytepair" else t["ink2"]
            flip = x > plot_x1 - 56
            tx = x - 10 if flip else x + 10
            anchor = "end" if flip else "start"
            o.append(f'<text x="{tx:.1f}" y="{cy+4}" '
                     f'font-family=\'{FONT}\' font-size="10.5" '
                     f'font-weight="{lw}" fill="{fill}" '
                     f'text-anchor="{anchor}" '
                     f'style="font-variant-numeric:tabular-nums">'
                     f'{fmt(v)}</text>')
        y0 += panel_h
    o.append("</svg>")
    return "\n".join(o)

def main():
    if len(sys.argv) != 4:
        print("usage: chart.py <results.json> <light.svg> <dark.svg>",
              file=sys.stderr)
        return 2
    run = json.load(open(sys.argv[1]))
    open(sys.argv[2], "w").write(svg(run, "light"))
    open(sys.argv[3], "w").write(svg(run, "dark"))
    print(f"wrote {sys.argv[2]} and {sys.argv[3]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
