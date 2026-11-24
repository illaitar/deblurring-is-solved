"""Progress of the measurement program, for the top of the site and of the README.

A measurement is one (method variant, test set, metric) cell. results/progress_cells.tsv lists every possible cell
(variants x the test sets they can run on x the ten metrics of the site) with the time its scores were complete
(modification time of the result file, UTC) and the time the variant's first result appeared. At a time t a cell is
possible once its variant, its test set and its metric all exist at t (a test set or a metric exists from its first
finished cell), and done once its scores are complete. Standard library only.
"""
import bisect
import csv
from datetime import datetime, timedelta, timezone

SET_LABEL = {"gopro": "GoPro", "reds_val": "REDS", "bsd2": "BSD", "dvd": "DVD", "realblur_j": "RealBlur-J",
             "realblur_r": "RealBlur-R", "gsblur": "GS-Blur", "miore": "MIORe"}
FONT = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
PALETTE = {
    "light": {"ink": "#0d0e11", "text": "#4a4c52", "muted": "#6e7178", "grid": "#ebecef", "fill": "#f1f2f4", "poss": "#9a9da4",
              "blue": "#2a78d6"},
    "dark": {"ink": "#f0f2f5", "text": "#c3c6cc", "muted": "#9095a0", "grid": "#2a2e35", "fill": "#1c2027", "poss": "#7d828c",
             "blue": "#6aa8f0"},
    # inline in the site: the page's own variables, so the charts follow its theme switch
    "css": {"ink": "var(--ink)", "text": "var(--ink-72)", "muted": "var(--ink-56)", "grid": "var(--ink-8)", "fill": "var(--paper-2)",
            "poss": "var(--ink-40)", "blue": "var(--img)"},
}


def _fi(c):
    return f'style="fill:{c}"'


def _sk(c):
    return f'style="stroke:{c}"'


def _t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if s else None


def compute(path, keys, base_of, now=None):
    """Series at every hour from the first finished cell to `now`, and the totals at `now`."""
    now = now or datetime.now(timezone.utc).replace(microsecond=0)
    cells = [c for c in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t") if c["key"] in keys]
    if not cells:
        return None
    first = {}
    for c in cells:
        first.setdefault(c["key"], _t(c["first"]) or now)
    set_t, met_t, done = {}, {}, []
    for c in cells:
        d = _t(c["done"])
        if d:
            set_t[c["set"]] = min(set_t.get(c["set"], d), d)
            met_t[c["metric"]] = min(met_t.get(c["metric"], d), d)
            done.append(d)
    poss = sorted(max(first[c["key"]], set_t.get(c["set"], now), met_t.get(c["metric"], now)) for c in cells)
    done.sort()
    order = sorted(first.items(), key=lambda kv: kv[1])
    t = min(done or [now]).replace(minute=0, second=0)
    grid = []
    while t < now:
        grid.append(t)
        t += timedelta(hours=1)
    grid.append(now)
    series, i, seen_k, seen_b = [], 0, 0, set()
    for t in grid:
        while i < len(order) and order[i][1] <= t:
            seen_b.add(base_of.get(order[i][0], order[i][0]))
            seen_k += 1
            i += 1
        series.append((t, bisect.bisect_right(poss, t), bisect.bisect_right(done, t), len(seen_b), seen_k))
    by_set = {}
    for c in cells:
        s = by_set.setdefault(c["set"], [0, 0])
        s[1] += 1
        s[0] += bool(c["done"])
    last = series[-1]
    return {"series": series, "possible": last[1], "done": last[2], "methods": last[3], "variants": last[4],
            "sets": len(by_set), "metrics": len({c["metric"] for c in cells}),
            "by_set": {k: by_set[k] for k in SET_LABEL if k in by_set}, "first": series[0][0], "now": now}


def _nice(v):
    for e in range(0, 7):                # smallest step of 1, 2, 2.5 or 5 x 10^e whose four gridlines cover v
        for m in (1, 2, 2.5, 5):
            n = m * 10 ** e
            if n * 4 >= v:
                return n
    return v / 4


def _day_ticks(t0, t1):
    t = (t0 + timedelta(days=1)).replace(hour=0, minute=0, second=0)
    out = []
    while t <= t1:
        out.append(t)
        t += timedelta(days=1)
    return out


def _chart(series, lines, title, legend, pal, W, H, x0=0, y0=0):
    """lines: [(index into series tuples, colour, width, dash, area?)]; legend: [(colour, dashed, text)]."""
    ml, mr, mt, mb = 48, 10, 52, 26
    t0, t1 = series[0][0], series[-1][0]
    span = max((t1 - t0).total_seconds(), 1.0)
    vmax = max(max(s[i] for s in series) for i, *_ in lines)
    step = _nice(vmax)
    ymax = step * 4
    X = lambda t: ml + (t - t0).total_seconds() / span * (W - ml - mr)  # noqa: E731
    Y = lambda v: mt + (1 - v / ymax) * (H - mt - mb)  # noqa: E731
    o = [f'<g transform="translate({x0} {y0})" font-family="{FONT}">',
         f'<text x="0" y="15" font-size="13" font-weight="600" {_fi(pal["ink"])}>{title}</text>']
    lx = 0
    for col, dashed, txt in legend:
        o.append(f'<line x1="{lx}" y1="33" x2="{lx + 16}" y2="33" {_sk(col)} stroke-width="2.5"' + (' stroke-dasharray="4 3"' if dashed else "") + "/>")
        o.append(f'<text x="{lx + 21}" y="37" font-size="11.5" {_fi(pal["text"])}>{txt}</text>')
        lx += 27 + 6.4 * len(txt)
    for k in range(5):
        v = step * k
        o.append(f'<line x1="{ml}" y1="{Y(v):.1f}" x2="{W - mr}" y2="{Y(v):.1f}" {_sk(pal["grid"])} stroke-width="1"/>')
        o.append(f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" font-size="10.5" text-anchor="end" {_fi(pal["muted"])}>{int(v):,}</text>')
    for t in _day_ticks(t0, t1):
        o.append(f'<line x1="{X(t):.1f}" y1="{H - mb}" x2="{X(t):.1f}" y2="{H - mb + 4}" {_sk(pal["poss"])} stroke-width="1"/>')
        o.append(f'<text x="{X(t):.1f}" y="{H - mb + 16}" font-size="10.5" text-anchor="middle" {_fi(pal["muted"])}>{t.day} {t.strftime("%b")}</text>')
    for i, col, wdt, dash, area in lines:
        pts = " ".join(f"{X(s[0]):.1f},{Y(s[i]):.1f}" for s in series)
        if area:
            o.append(f'<polygon points="{X(t0):.1f},{Y(0):.1f} {pts} {X(t1):.1f},{Y(0):.1f}" {_fi(area)}/>')
        o.append(f'<polyline points="{pts}" fill="none" {_sk(col)} stroke-width="{wdt}" stroke-linejoin="round"'
                 + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
    o.append(f'<line x1="{ml}" y1="{Y(0):.1f}" x2="{W - mr}" y2="{Y(0):.1f}" {_sk(pal["poss"])} stroke-width="1"/>')
    o.append("</g>")
    return "".join(o)


def svgs(p, theme="light"):
    """(measurements chart, methods chart, both side by side) as SVG strings."""
    pal, s = PALETTE[theme], p["series"]
    pct = 100 * p["done"] / p["possible"] if p["possible"] else 0
    m = _chart(s, [(1, pal["poss"], 1.6, "5 4", pal["fill"]), (2, pal["ink"], 2.2, None, None)],
               "Measurements: method variant × test set × metric",
               [(pal["ink"], False, f"done {p['done']:,} ({pct:.0f} %)"), (pal["poss"], True, f"possible {p['possible']:,}")],
               pal, 480, 230)
    v = _chart(s, [(4, pal["poss"], 1.8, None, None), (3, pal["blue"], 2.2, None, None)], "Methods in the benchmark",
               [(pal["blue"], False, f"methods {p['methods']}"), (pal["poss"], False, f"variants {p['variants']}")], pal, 280, 230)
    alt = f"{p['done']:,} of {p['possible']:,} measurements done; {p['methods']} methods, {p['variants']} variants"
    one = lambda body, w: (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 230" role="img" aria-label="{alt}">'  # noqa: E731
                           f'<title>{alt}</title>{body}</svg>')
    both = one(m + f'<g transform="translate(500 0)">{v}</g>', 780)
    return one(m, 480), one(v, 280), both
