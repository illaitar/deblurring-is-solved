"""Build the project site and the README tables from the result tables.

    python tools/build_site.py                 # docs/index.html + docs/leaderboard.csv (GitHub Pages, /docs folder)
    python tools/build_site.py --fragment F    # also write the page without the <html>/<head>/<body> wrapper

Inputs (all under results/):
    master.json               one row per method and variant: per-set PSNR / SSIM / LPIPS, frames, status, size
    per_sequence.tsv          per-sequence scores of the same cells (key, set, sequence, frames, psnr, ssim, lpips)
    input_per_sequence.tsv    PSNR of the untouched blurry input per sequence (set, sequence, frames, psnr_in)

The page template is site/template.html. {{name}} placeholders are filled from FACTS, and the JSON payload replaces
/*__DATA__*/null. The README block between <!-- generated:start --> and <!-- generated:end --> is rewritten here too.
"""
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SETS = ["gopro", "reds_val", "bsd2", "dvd", "realblur_j", "realblur_r", "gsblur", "miore"]
SET_LABEL = {"gopro": "GoPro", "reds_val": "REDS", "bsd2": "BSD", "dvd": "DVD", "realblur_j": "RealBlur-J",
             "realblur_r": "RealBlur-R", "gsblur": "GS-Blur", "miore": "MIORe"}
FULL = {"gopro": 1111, "reds_val": 3000, "bsd2": 3000, "dvd": 1000, "realblur_j": 980, "realblur_r": 980, "gsblur": 1000,
        "miore": 300}
VIDEO_SETS = SETS[:4]          # RealBlur, GS-Blur and MIORe hold single images: image methods only
# Mean PSNR of the blurry input over all protocol frames (frame-weighted, as the method scores are). RealBlur: the
# official evaluation (ECC alignment to the sharp image, as for the methods), filled from rb_cells.tsv when present.
INPUT_PSNR = {"gopro": 25.6401, "reds_val": 26.2657, "bsd2": 26.6321, "dvd": 27.2856}
TRAIN_LABEL = {"GoPro": "GoPro", "AiO-mix": "all-in-one mix", "GoPro+DVD+NFS": "GoPro + DVD + NFS",
               "DVD": "DVD", "REDS": "REDS", "BSD1": "BSD 1ms/8ms", "BSD2": "BSD 2ms/16ms", "BSD3": "BSD 3ms/24ms",
               "RealBlurJ": "RealBlur-J", "RealBlurR": "RealBlur-R", "RSBlur": "RSBlur", "HIDE": "HIDE", "NAH": "NAH",
               "REDS_ours": "REDS (retrained by us)", "GoPro_ours": "GoPro (retrained by us)",
               "RealBlur": "RealBlur (subset not stated)", "GoProRealBlurJ": "GoPro + RealBlur-J", "GoProREDS": "GoPro + REDS", "GoProBSDBRealBlurJ": "GoPro + BSD-B + RealBlur-J",
               "GoProBSDBRealBlurR": "GoPro + BSD-B + RealBlur-R", "GoProRealBlurR": "GoPro + RealBlur-R",
               "GoProRealBlur": "GoPro + RealBlur (inferred, not stated by the authors)",
               "REDSjpeg": "REDS with JPEG (NTIRE 2021)", "BSD": "BSD (exposure not stated)"}
# The "Trained on" filter of the site offers single datasets only. Every exact training label belongs to at most one of
# them. A checkpoint trained on a mix of sets, or on a set that is not listed, has no group and shows under "any set".
TRAIN_GROUP = {"GoPro": "GoPro", "REDS": "REDS", "REDSjpeg": "REDS", "REDS_ours": "REDS", "BSD": "BSD", "BSD1": "BSD",
               "BSD2": "BSD", "BSD3": "BSD", "DVD": "DVD", "RealBlurJ": "RealBlur", "RealBlurR": "RealBlur", "RealBlur": "RealBlur"}
GROUP_ORDER = ["GoPro", "REDS", "BSD", "DVD", "RealBlur"]
# (row, test set) whose test clips were part of the checkpoint's training data: the REDS models of BasicSR (EDVR) and
# VRT 007 were trained on REDS train + val without REDS4, and our REDS test set is the official val split.
SEEN = {("EDVR-REDS", "reds_val"), ("VRT@REDS", "reds_val")}
# training label -> the test set it is in-domain for (the "trained on the test set" filter)
IN_DOMAIN = {"GoPro": "gopro", "REDS": "reds_val", "REDS_ours": "reds_val", "REDSjpeg": "reds_val", "BSD": "bsd2", "BSD1": "bsd2", "BSD2": "bsd2", "BSD3": "bsd2",
             "DVD": "dvd", "RealBlurJ": "realblur_j", "RealBlurR": "realblur_r", "GoPro_ours": "gopro"}
SITE_URL = "https://illaitar.github.io/deblurring-is-solved/"   # the GitHub Pages address (docs/ of the default branch)
AUTHORS = ["Nikita Alutis", "Egor Chistov"]
PYIQA = "0.1.16"  # envs/iqa/FREEZE.txt on the cluster

args = sys.argv[1:]
FRAGMENT = args[args.index("--fragment") + 1] if "--fragment" in args else None
# Viewer bundles (img/<set>/<n>.bin, about 230 MB) are hosted outside the repository when --img-base is given, e.g.
# https://huggingface.co/datasets/<owner>/<repo>/resolve/main/ ; the page then keeps only img/index.json.
IMG_BASE = args[args.index("--img-base") + 1] if "--img-base" in args else ""


def num(x, nd):
    return None if x is None or x == "" else round(float(x), nd)


# ---------------------------------------------------------------- rows
master = json.load(open(RES / "master.json", encoding="utf-8"))

seq_in = {}
for r in csv.DictReader((ln for ln in open(RES / "input_per_sequence.tsv", encoding="utf-8") if not ln.startswith("#")),
                        delimiter="\t"):
    seq_in.setdefault(r["sub"], {})[r["sequence"]] = float(r["psnr_in"])

seq_scores = {}
for r in csv.DictReader(open(RES / "per_sequence.tsv", encoding="utf-8"), delimiter="\t"):
    seq_scores.setdefault((r["key"], r["set"]), {})[r["sequence"]] = float(r["psnr"])

# Extra image-quality metrics (pyiqa, newbl/iqa on the cluster): iqa_cells.tsv per method and set, iqa_ref.tsv for the
# sharp frames (gt), the blurry input (input: no-reference metrics, input_fr: full-reference metrics against gt).
# A cell enters the page only when every stored frame of the run has been scored ("complete" = 1).
IQA = {"ms_ssim": "ms_ssim", "lpips-vgg": "lpips_vgg", "dists": "dists",
       "paq2piq": "paq2piq", "musiq": "musiq", "clipiqa+": "clipiqa", "maniqa": "maniqa", "based4": "based4"}
iqa_cell, iqa_ref, iqa_have = {}, {}, set()
# BASED-4 (based-metric package, restored frame against its blurry input) comes in the same two-table layout
for cells_f, ref_f in (("iqa_cells.tsv", "iqa_ref.tsv"), ("based_cells.tsv", "based_ref.tsv")):
    if (RES / cells_f).exists():
        for r in csv.DictReader(open(RES / cells_f, encoding="utf-8"), delimiter="\t"):
            if r["complete"] != "1":
                continue
            v = {js: round(float(r[k]), 4) for k, js in IQA.items() if r.get(k)}
            iqa_cell.setdefault((r["key"], r["set"]), {}).update(v)
            iqa_have |= set(v)
    if (RES / ref_f).exists():
        for r in csv.DictReader(open(RES / ref_f, encoding="utf-8"), delimiter="\t"):
            full_set = int(r["frames"]) == FULL[r["set"]]
            kind = "gt" if r["kind"] == "gt" else "input"
            if full_set:
                v = {js: round(float(r[k]), 4) for k, js in IQA.items() if r.get(k)}
                if r["kind"] == "input_fr" and r.get("lpips"):   # pyiqa lpips = the same AlexNet LPIPS as the runs
                    v["lpips"] = round(float(r["lpips"]), 4)
                iqa_ref.setdefault(r["set"], {}).setdefault(kind, {}).update(v)
# official RealBlur evaluation of the blurry input (rb_cells.tsv, dir _ref): PSNR and SSIM after ECC alignment
if (RES / "rb_cells.tsv").exists():
    for r in csv.DictReader(open(RES / "rb_cells.tsv", encoding="utf-8"), delimiter="\t"):
        if r["dir"] == "_ref" and int(r["images"]) == FULL[r["set"]]:
            INPUT_PSNR[r["set"]] = round(float(r["psnr"]), 4)
            iqa_ref.setdefault(r["set"], {}).setdefault("input", {})["ssim"] = round(float(r["ssim"]), 4)
# GS-Blur and MIORe (26.11): the mean over the test images of the blurry input's PSNR, from the frame table
# (results/frames/data/frames_<set>.json, column psnr_in, computed as eval/score.py does for the methods)
for ds in ("gsblur", "miore"):
    _ff = RES / "frames" / "data" / f"frames_{ds}.json"
    if _ff.exists():
        _fj = json.loads(_ff.read_text(encoding="utf-8"))
        _v = [x[_fj["cols"].index("psnr_in")] for x in _fj["frames"] if x[_fj["cols"].index("psnr_in")] is not None]
        if len(_v) == FULL[ds]:
            INPUT_PSNR[ds] = round(sum(_v) / len(_v), 4)

seq_names = {}
for ds in SETS:
    names = set(seq_in.get(ds, {}))
    for (k, d), v in seq_scores.items():
        if d == ds:
            names |= set(v)
    seq_names[ds] = sorted(names) if ds in VIDEO_SETS else []   # RealBlur: 980 single images, no per-sequence strip

rows = []
for r in master:
    row = {
        "key": r["key"], "base": r["base"], "name": r["name"], "variant": r["variant"], "task": r["task"],
        "family": r["cluster"], "year": int(r["year"]) if str(r["year"]).isdigit() else None,
        "venue": r["venue"] if r["venue"] not in ("-", "unknown") else "",
        "train": TRAIN_LABEL.get(r["train"], r["train"]), "tr": r["train"], "tg": TRAIN_GROUP.get(r["train"], ""),
        "gopro_trained": r["train"] == "GoPro",
        "dom": IN_DOMAIN.get(r["train"], ""),
        "params": num(r["params_M"], 3), "gmac": num(r["gmac"], 1), "ms": num(r["ms_720p"], 2), "s": {},
    }
    for ds in SETS:
        st = r.get(f"{ds}_status")
        if st not in ("ok", "partial") or r[f"{ds}_psnr"] is None:
            continue
        cell = {"psnr": num(r[f"{ds}_psnr"], 4), "ssim": num(r[f"{ds}_ssim"], 4), "lpips": num(r[f"{ds}_lpips"], 4),
                "n": r[f"{ds}_n"], "ok": st == "ok"}
        cell.update(iqa_cell.get((r["key"], ds), {}))
        if (r["key"], ds) in SEEN:
            cell["seen"] = 1
        sq = seq_scores.get((r["key"], ds))
        if sq:
            cell["seq"] = [round(sq[s], 3) if s in sq else None for s in seq_names[ds]]
        row["s"][ds] = cell
    rows.append(row)

def sets_of(r):
    return SETS if r["task"] == "image" else VIDEO_SETS


complete_all = [r for r in rows if all(ds in r["s"] and r["s"][ds]["ok"] for ds in sets_of(r))]
some = [r for r in rows if r["s"] and r not in complete_all]
none_yet = [r for r in rows if not r["s"]]
years = [int(r["year"]) for r in rows if str(r["year"]).isdigit()]
nr_names = [n for k, n in (("paq2piq", "PaQ-2-PiQ"), ("musiq", "MUSIQ"), ("clipiqa", "CLIP-IQA+"), ("maniqa", "MANIQA")) if k in iqa_have]
FACTS = {
    "n_rows": len(rows), "n_complete": len(complete_all), "n_some": len(some), "n_none": len(none_yet),
    "n_methods": len({r["base"] for r in master}), "n_sets": len(SETS),
    "n_ckpt": len({(r["base"], r["train"]) for r in master}),
    "frames": f"{sum(FULL.values()):,}",
    "year0": min(years), "year1": max(years), "built": date.today().isoformat(),
    "authors": ", ".join(AUTHORS[:-1]) + " and " + AUTHORS[-1],
    "authors_html": "".join(f'<span class="au">{a}</span>' for a in AUTHORS),
    "iqa_note": ("<li>The perceptual and no-reference scores (MS-SSIM, LPIPS-VGG, DISTS, " + ", ".join(nr_names)
                 + ") are the defaults of pyiqa " + PYIQA
                 + ", computed in fp32 on the stored lossless outputs of each run. A row shows them once every frame of"
                 " the run is scored.</li>") if iqa_have - {"based4"} else "",
    "based_note": ("<li>BASED-4 (higher is better) scores the restored frame against its own blurry input, with no sharp"
                   " reference: four convolutional features and a linear head fitted to a subjective study of"
                   " deblurring results (BASED, VCIP 2023). It is computed on the stored lossless outputs.</li>")
                  if "based4" in iqa_have else "",
}

# ---------------------------------------------------------------- progress of the measurement program
# results/progress_cells.tsv: every possible (variant, test set, metric) cell and when it was done (tools/progress.py).
# Shown at the top of the site and of the README, as the numbers of the moment and their history.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import progress as PROG  # noqa: E402

PG = PROG.compute(RES / "progress_cells.tsv", {r["key"] for r in rows}, {r["key"]: r["base"] for r in master}) \
    if (RES / "progress_cells.tsv").exists() else None
# The best GoPro-trained row on GoPro (for the dashboard) and the GitHub stars at build time (the page refreshes them)
_bg = max((r for r in rows if r["gopro_trained"] and r["s"].get("gopro", {}).get("ok")), key=lambda r: r["s"]["gopro"]["psnr"], default=None)
BEST = (_bg["s"]["gopro"]["psnr"], _bg["name"], _bg["task"]) if _bg else (0.0, "", "")
REPO = "illaitar/deblurring-is-solved"
STARS = None
try:
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{REPO}",
                                                       headers={"User-Agent": "dbis-site-build"}), timeout=10) as _r:
        STARS = int(json.loads(_r.read().decode("utf-8")).get("stargazers_count"))
except Exception:
    STARS = None
if PG:
    _tiles = PROG.tiles(PG, BEST, len(SETS), "css")
    _tile = lambda v, l, s, sp: f'<div class="tile"><span class="v">{v}</span><span class="l">{l}</span><span class="s">{s}</span>{sp}</div>'  # noqa: E731
    _star = ('<a class="tile star" href="https://github.com/' + REPO + '" target="_blank" rel="noopener">'
             '<span class="v"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.6l1.9 4 4.3.5-3.2 2.9.9 4.3L8 11.1 4.1 13.3l.9-4.3L1.8 6.1l4.3-.5z" '
             'fill="currentColor"/></svg><span data-stars>' + (f"{STARS:,}" if STARS is not None else "–") + '</span></span>'
             '<span class="l">stars on GitHub</span><span class="s">' + REPO + '</span></a>')
    FACTS["progress_html"] = (
        '<div class="dash" aria-label="Key numbers">' + "".join(_tile(*x) for x in _tiles[:2]) + _star + "".join(_tile(*x) for x in _tiles[2:])
        + f'<p class="dash-n">A measurement is one method variant on one test set with one metric. The two small charts follow the '
          f'result files hour by hour, up to {PG["now"].strftime("%d %b %Y, %H:%M")} UTC.</p></div>')
else:
    FACTS["progress_html"] = ""
# the GitHub stars after the authors' names (24.11: the tiles are off the page for now; progress_html is kept for later)
FACTS["stars_html"] = (f'<a class="stars" href="https://github.com/{REPO}" target="_blank" rel="noopener" title="Stars of the repository on GitHub">'
                       '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.6l1.9 4 4.3.5-3.2 2.9.9 4.3L8 11.1 4.1 13.3l.9-4.3L1.8 6.1l4.3-.5z" '
                       'fill="currentColor"/></svg><span data-stars>' + (f"{STARS:,}" if STARS is not None else "–") + '</span>'
                       '<span class="sr"> stars on GitHub</span></a>')

# ---------------------------------------------------------------- the mark
# site/logo-inline.svg takes its colours from the page's CSS variables (it follows the theme button); __ID__ keeps the
# filter ids of its two copies apart. site/logo.svg, logo-dark.svg (README) and favicon.svg are copied to docs/.
# The d of both wordmarks is the real d of Geologica (600 in the bar, 700 in the title) with the shake and the iris
# drawn on it: site/dletter-<weight>.svg, made by logo/glyph_d.py from the font files the site loads.
_dl = lambda w: (ROOT / "site" / f"dletter-{w}.svg").read_text(encoding="utf-8") if (ROOT / "site" / f"dletter-{w}.svg").exists() else ""  # noqa: E731
FACTS["logo_bar"] = _dl(600).replace("__ID__", "lb")
FACTS["logo_hero"] = _dl(700).replace("__ID__", "lh")

# ---------------------------------------------------------------- dataset cards
# Static facts and citations: site/datasets.json (curated from the papers and dataset pages, see datasets_sources.json).
# Measured characteristics, the same for every set: medians over the test frames of results/frames/data/frames_<set>.json.
CARD_MEASURED = ["grad_ratio", "motion", "texture", "brightness", "highlights", "noise", "psnr_med"]
cards = json.loads((ROOT / "site" / "datasets.json").read_text(encoding="utf-8")) if (ROOT / "site" / "datasets.json").exists() else {}
for ds in SETS:
    f = RES / "frames" / "data" / f"frames_{ds}.json"
    meas = {}
    if f.exists():
        fj = json.loads(f.read_text(encoding="utf-8"))
        cols, fr = fj["cols"], fj["frames"]
        for c in CARD_MEASURED:
            if c in cols:
                v = sorted(x[cols.index(c)] for x in fr if x[cols.index(c)] is not None)
                if v:
                    meas[c] = round(v[len(v) // 2], 3)
        meas["sequences"] = len({x[0] for x in fr})
    cards.setdefault(ds, {})["measured"] = meas
# Collage of each card: site/collage/<set>.jpg, listed in site/collage/index.json with the frames it shows. Eight blurry
# test frames per set at evenly spaced quantiles of the input PSNR (1/16, 3/16, ..., 15/16), one per sequence
# (RealBlur: per scene), ordered from the strongest to the weakest blur.
COLLAGE = ROOT / "site" / "collage"
if (COLLAGE / "index.json").exists():
    for ds, c in json.loads((COLLAGE / "index.json").read_text(encoding="utf-8")).items():
        if ds in cards and (COLLAGE / c["file"]).exists():
            cards[ds]["collage"] = {"src": "collage/" + c["file"], "w": c["w"], "h": c["h"],
                                    "lo": c["frames"][0]["psnr_in"], "hi": c["frames"][-1]["psnr_in"],
                                    "unit": ("scene" if ds.startswith("realblur") or ds == "gsblur"
                                             else "image" if ds == "miore" else "sequence")}

# ---------------------------------------------------------------- 25.11 layout: tags, data cards, dbis numbers
TAGS = ROOT / "site" / "tags"


def _tag(name):
    return (TAGS / name).read_text(encoding="utf-8").replace("<svg ", '<svg class="dtag" ', 1)


FACTS["tag_dbis"], FACTS["tag_dbisf"] = _tag("dbis-plain-css.svg"), _tag("dbisf-plain-css.svg")
FACTS["tag_dbis_big"], FACTS["tag_dbisf_big"] = _tag("dbis-css.svg"), _tag("dbisf-css.svg")
DBIS_FILES = ["NAFNet-64.json", "BSSTNet.json", "DSTNetPlus-L.json"]      # results/dbis/<file>, each with a "site" block
DBIS_URL = "https://huggingface.co/Illaitar/dbis/resolve/main/"   # the hosted weight files (dbis.URL); None: "soon"
MINUS = "−"
DBS = [json.loads((RES / "dbis" / f).read_text(encoding="utf-8")) | {"_file": f} for f in DBIS_FILES]
def _faster(a, x):
    """The time of dbisf against the authors': "24 % faster" (less time) below 1.5 times, else "5.6× faster"."""
    return f"{a / x:.1f}&times; faster" if a / x >= 1.5 else f"{100 * (1 - x / a):.0f}&nbsp;% faster"


FACTS["dbisf_gains"] = ", ".join(       # the "What you get" row: time and weight size of every dbisf against the authors
    f'{_J["site"]["title"].split(",")[0]} <b>{_faster(_J["site"]["latency_ms"]["authors"], _J["site"]["latency_ms"]["dbisf"])}, weights '
    f'{_J["site"]["weights_bytes"]["authors"] / _J["site"]["weights_bytes"]["dbisf"]:.1f}&times; smaller</b>'
    for _J in DBS)


def _t(ms):
    return f"{ms:.1f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def _times(a, x, word):
    q = a / x
    return ["same", False] if abs(q - 1) < .05 else [f"{q:.1f}× {word}", q > 1]


def _pct(x, a):
    p = 100 * (x / a - 1)
    return ["same", False] if abs(p) < 1 else [f"{MINUS if p < 0 else '+'}{abs(p):.0f} %", p < 0]


def _dpsnr(d):
    return ["same", False] if abs(d) < 5e-6 else [f"{MINUS if d < 0 else '+'}{abs(d):.5f} dB", False]


_rows = []
for _J in DBS:
    _s = _J["site"]
    _L, _W, _T, _P = _s["latency_ms"], _s["weights_bytes"], _s["load_s"], _s["gopro_psnr"]
    _rows.append(f'<tr class="grp"><th scope="rowgroup" colspan="5">{_s["title"]}</th></tr>')
    for _kd, _lab in (("authors", "authors' code"), ("dbis", FACTS["tag_dbis"]), ("dbisf", FACTS["tag_dbisf"])):
        _b = (lambda v: f"<b>{v}</b>") if _kd == "dbisf" else (lambda v: v)
        _rows.append(f'<tr><td>{_lab}</td><td>{_b(f"{_W[_kd] / 1e6:.0f} MB")}</td><td>{_T[_kd]:.2f} s</td>'
                     f'<td>{_b(_t(_L[_kd]))}</td><td>{_P[_kd]:.4f}</td></tr>')
FACTS["dbis_table"] = (
    '<div class="scroll whybox"><table class="dbt"><thead><tr><th scope="col">Model</th><th scope="col">Weights</th>'
    '<th scope="col">Load</th><th scope="col">Time per 720p frame</th><th scope="col">GoPro PSNR</th></tr></thead><tbody>'
    + "".join(_rows) + '</tbody></table></div><p class="fig-note">Load: a fresh process, from CUDA ready to the network with '
    'its weights on the GPU. Time: the benchmark\'s latency harness on one A100, 1280&times;720, batch 1, the three versions '
    'of a method on the same node. ' + " ".join(f'{J["site"]["title"].split(",")[0]}: {J["site"]["time_mode"]}.' for J in DBS) +
    ' PSNR: all 1,111 GoPro test frames. Every number: ' + ", ".join(f'<a href="https://github.com/illaitar/deblurring-is-solved/blob/main/results/dbis/{J["_file"]}">'
                                  f'results/dbis/{J["_file"]}</a>' for J in DBS) + '.</p>')
DBIS_PAY = {}
for _J in DBS:
    _s = _J["site"]
    _L, _W, _T, _P = _s["latency_ms"], _s["weights_bytes"], _s["load_s"], _s["gopro_psnr"]
    DBIS_PAY[_s["row_key"]] = {
        "source": _s["source"], "checkpoint": _s["checkpoint"], "note": _s["note"], "extra_note": _s.get("extra_note"),
        "pip": _s.get("pip", "torch numpy safetensors zstandard"),
        "authors": {"mb": round(_W["authors"] / 1e6), "ms": _L["authors"], "load": _T["authors"], "psnr": _P["authors"]},
        "models": [{"name": _s["models"][kd], "kind": kd, "ext": _s["ext"][kd], "file": _s["models"][kd] + _s["ext"][kd],
                    "about": _s.get("about", {}).get(kd), "mb": round(_W[kd] / 1e6), "ms": _L[kd], "load": _T[kd],
                    "psnr": _P[kd], "d_mb": _times(_W["authors"], _W[kd], "smaller"), "d_ms": _times(_L["authors"], _L[kd], "faster") if _L["authors"] / _L[kd] >= 1.5 else _pct(_L[kd], _L["authors"]),
                    "d_load": _times(_T["authors"], _T[kd], "faster"), "d_psnr": _dpsnr(_P[kd] - _P["authors"]),
                    "url": DBIS_URL + _s["models"][kd] + _s["ext"][kd] if DBIS_URL else None} for kd in ("dbis", "dbisf")]}
REPOS = json.loads((ROOT / "site" / "method_repos.json").read_text(encoding="utf-8"))
DATA_CARDS = [  # set, name, pairs, size, blur, license, authors' page
    ("gopro", "GoPro", "1,111 pairs", "1280&times;720", "", "CC BY 4.0", "https://seungjunnah.github.io/Datasets/gopro.html"),
    ("reds_val", "REDS val", "3,000 pairs", "1280&times;720", "", "CC BY 4.0", "https://seungjunnah.github.io/Datasets/reds.html"),
    ("bsd2", "BSD 2ms/16ms", "3,000 pairs", "640&times;480", "real blur", "no license stated", "https://github.com/zzh-tech/ESTRNN"),
    ("dvd", "DVD", "1,000 pairs", "1280&times;720", "", "no license stated", "https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/"),
    ("realblur_j", "RealBlur-J", "980 images", "&asymp;670&times;760", "real blur", "CC BY 4.0", "https://github.com/rimchang/RealBlur"),
    ("realblur_r", "RealBlur-R", "980 images", "&asymp;680&times;773", "real blur", "CC BY 4.0", "https://github.com/rimchang/RealBlur"),
    ("gsblur", "GS-Blur", "1,000 images", "480&times;270 to 1920&times;1080", "", "no license stated", "https://github.com/dongwoohhh/GS-Blur"),
    ("miore", "MIORe", "300 images", "1920&times;1080", "", "no license stated", "https://github.com/george200150/MIORe"),
]
_cards = []
for ds, nm, n, size, blur, lic, url in DATA_CARDS:
    col = cards.get(ds, {}).get("collage")
    img = (f'<img src="{col["src"]}" alt="Blurry frames of {nm}" loading="lazy">' if col else '<span class="ph">being scored</span>')
    inp = f"input {INPUT_PSNR[ds]:.2f} dB" if ds in INPUT_PSNR and INPUT_PSNR[ds] is not None else "scores coming"
    facts = " &middot; ".join(x for x in (size, inp, blur) if x)
    pick = (f'<button type="button" class="lk" data-pick="{ds}">Leaderboard</button> &middot; ' if ds in SETS else "")
    _cards.append(f'<div class="dc"><div class="top"><span class="nm">{nm}</span><span class="ct">{n}</span></div>{img}'
                  f'<span class="fx">{facts}</span><span class="ln">{pick}<a href="{url}" target="_blank" rel="noopener">authors\' release</a> &middot; {lic}</span></div>')
FACTS["data_cards"] = "".join(_cards)

data = {
    "datasets": cards, "imgBase": IMG_BASE,
    "sets": SETS, "setLabel": SET_LABEL, "full": FULL, "input": INPUT_PSNR,
    "seq": seq_names, "seqInput": {ds: [round(seq_in[ds][s], 3) if s in seq_in.get(ds, {}) else None
                                        for s in seq_names[ds]] for ds in SETS},
    "rows": rows, "extra": sorted(iqa_have), "ref": iqa_ref, "videoSets": VIDEO_SETS,
    "trainGroups": [g for g in GROUP_ORDER if g != "GoPro" and any(r["tg"] == g for r in rows)],
    "dbis": {k: v for k, v in DBIS_PAY.items() if any(r["key"] == k for r in rows)},
    "repo": {k: u for k, u in REPOS.items() if any(k in (r["base"], r["key"]) for r in rows)},   # base, or a row's key
    "tags": {"dbis": FACTS["tag_dbis"], "dbisf": FACTS["tag_dbisf"]},
}

# ---------------------------------------------------------------- page
tpl = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(FACTS))
if missing:
    sys.exit(f"template placeholders without a value: {missing}")
page = re.sub(r"\{\{(\w+)\}\}", lambda m: str(FACTS[m.group(1)]), tpl)
payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
page = page.replace("/*__DATA__*/null", payload)
# frame-level sections (site/frames.js) and their lazily fetched data: results/frames/{data,img} -> docs/{data,img}
page = page.replace("/*__FRAMES_JS__*/", (ROOT / "site" / "frames.js").read_text(encoding="utf-8").replace("</script", "<\\/script"))
import shutil  # noqa: E402
src = RES / "frames" / "data"
if src.is_dir():
    shutil.copytree(src, ROOT / "docs" / "data", dirs_exist_ok=True)
src = RES / "frames" / "img"
if src.is_dir():
    if IMG_BASE:
        (ROOT / "docs" / "img").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "index.json", ROOT / "docs" / "img" / "index.json")
    else:
        shutil.copytree(src, ROOT / "docs" / "img", dirs_exist_ok=True)
out = ROOT / "docs"
out.mkdir(exist_ok=True)
if COLLAGE.is_dir():
    (out / "collage").mkdir(exist_ok=True)
    for p in sorted(COLLAGE.glob("*.jpg")):
        shutil.copyfile(p, out / "collage" / p.name)
(out / "index.html").write_text(page, encoding="utf-8")
(out / ".nojekyll").write_text("", encoding="utf-8")
for name in ("logo.svg", "logo-dark.svg", "favicon.svg", "wordmark.svg", "wordmark-dark.svg"):
    if (ROOT / "site" / name).exists():
        shutil.copyfile(ROOT / "site" / name, out / name)
if PG:                                   # the dashboard of the README, one image per GitHub theme
    (out / "dashboard.svg").write_text(PROG.tiles_svg(PG, BEST, len(SETS), "light"), encoding="utf-8")
    (out / "dashboard-dark.svg").write_text(PROG.tiles_svg(PG, BEST, len(SETS), "dark"), encoding="utf-8")

with open(out / "leaderboard.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    head = ["method", "variant", "type", "family", "year", "venue", "trained_on", "params_M", "gmac_720p", "ms_720p"]
    xm = [js for js in IQA.values() if js in iqa_have]
    for ds in SETS:
        head += [f"{SET_LABEL[ds]}_{m}" for m in ["psnr", "ssim", "lpips"] + xm + ["frames", "final"]]
    w.writerow(head)
    for r in sorted(rows, key=lambda r: -(r["s"].get("gopro", {}).get("psnr") or 0)):
        line = [r["name"], r["variant"], r["task"], r["family"], r["year"], r["venue"], r["train"], r["params"], r["gmac"], r["ms"]]
        for ds in SETS:
            c = r["s"].get(ds)
            line += ([c["psnr"], c["ssim"], c["lpips"]] + [c.get(m, "") for m in xm] + [c["n"], "yes" if c["ok"] else "no"]
                     if c else [""] * (5 + len(xm)))
        w.writerow(line)


# ---------------------------------------------------------------- README: one table per test set
def md_table(ds, n=None):
    rs = [r for r in rows if r["gopro_trained"] and ds in r["s"] and r["s"][ds]["ok"]]
    rs.sort(key=lambda r: -r["s"][ds]["psnr"])
    lines = ["| # | Method | Type | Params, M | PSNR | SSIM | LPIPS |", "|---:|---|---|---:|---:|---:|---:|"]
    for i, r in enumerate(rs[:n]):
        c = r["s"][ds]
        meta = r["variant"].replace(",", ", ") + (f", {r['venue']} {r['year']}" if r["venue"] else f", {r['year']}")
        par = f"{r['params']:.1f}" if r["params"] else "–"
        f3 = lambda x: f"{x:.3f}" if x is not None else "–"
        lines.append(f"| {i + 1} | **{r['name']}** <sub>{meta}</sub> | {r['task']} | {par} | {c['psnr']:.2f} | {f3(c['ssim'])} | {f3(c['lpips'])} |")
    return "\n".join(lines), len(rs)


site_link = f"[interactive leaderboard]({SITE_URL})" if SITE_URL else "interactive leaderboard in [`docs/`](docs/) (GitHub Pages)"
blocks = []
for ds in SETS:
    top, n = md_table(ds, 10)
    full, _ = md_table(ds)
    blocks.append(f"### {SET_LABEL[ds]}\n\n{top}\n\n<details>\n<summary>All {n} rows on {SET_LABEL[ds]}</summary>\n\n{full}\n\n</details>")
gen = ("<!-- generated:start (tools/build_site.py writes this block; edit the script, not the text) -->\n"
       f"> [!NOTE]\n> Numbers as of {FACTS['built']}: {FACTS['n_complete']} of {FACTS['n_rows']} variants scored on all their sets (eight for image methods, four for video methods), "
       f"{FACTS['n_some']} on some. GoPro-trained weights (other training sets are on the site), PSNR in dB (RealBlur: official ECC-aligned PSNR), sorted by PSNR. The full table is in "
       f"[`docs/leaderboard.csv`](docs/leaderboard.csv) and on the {site_link}.\n\n"
       + "\n\n".join(blocks) + "\n<!-- generated:end -->")
readme = ROOT / "README.md"
txt = readme.read_text(encoding="utf-8")
a, b = txt.find("<!-- generated:start"), txt.find("<!-- generated:end -->")
if a >= 0 and b > a:
    txt = txt[:a] + gen + txt[b + len("<!-- generated:end -->"):]
else:
    print("README.md has no generated block; left unchanged")
# progress block at the top of the README, between <!-- progress:start --> and <!-- progress:end -->
a, b = txt.find("<!-- progress:start"), txt.find("<!-- progress:end -->")
if PG and a >= 0 and b > a:
    pgen = ("<!-- progress:start (tools/build_site.py writes this block; edit the script, not the text) -->\n"
            f'<p align="center"><a href="https://github.com/{REPO}/stargazers"><img src="https://img.shields.io/github/stars/{REPO}?style=flat&amp;label=stars" alt="GitHub stars"></a></p>\n\n'
            f'<p align="center"><b>{PG["done"]:,} of {PG["possible"]:,} measurements done ({100 * PG["done"] / PG["possible"]:.0f} %)</b>'
            f' &nbsp;&middot;&nbsp; one measurement = a method variant on a test set with a metric &nbsp;&middot;&nbsp; '
            f'{PG["methods"]} methods, {PG["variants"]} variants, {PG["sets"]} test sets, {PG["metrics"]} metrics, as of '
            f'{PG["now"].strftime("%Y-%m-%d")}</p>\n<!-- progress:end -->')
    txt = txt[:a] + pgen + txt[b + len("<!-- progress:end -->"):]
readme.write_text(txt, encoding="utf-8")

if FRAGMENT:
    frag = re.sub(r"<!doctype html>\s*|</?html[^>]*>\s*|</?head>\s*|</?body>\s*", "", page, flags=re.I)
    frag = re.sub(r'<meta (charset|name="viewport")[^>]*>\s*', "", frag)
    # the artifact preview serves no binary type: the frame data go there as <file>.b64.txt (tools/stage_preview.py)
    frag = frag.replace("<script>/* Two methods, frame by frame", "<script>window.DBIS_B64 = true;</script>\n<script>/* Two methods, frame by frame", 1)
    Path(FRAGMENT).write_text(frag, encoding="utf-8")

print(f"rows {len(rows)} (complete {len(complete_all)}, partial {len(some)}, none {len(none_yet)}), page {len(page) / 1024:.0f} KiB")
