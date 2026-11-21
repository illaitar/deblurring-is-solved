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
SETS = ["gopro", "reds_val", "bsd2", "dvd", "realblur_j", "realblur_r"]
SET_LABEL = {"gopro": "GoPro", "reds_val": "REDS", "bsd2": "BSD", "dvd": "DVD", "realblur_j": "RealBlur-J",
             "realblur_r": "RealBlur-R"}
FULL = {"gopro": 1111, "reds_val": 3000, "bsd2": 3000, "dvd": 1000, "realblur_j": 980, "realblur_r": 980}
VIDEO_SETS = SETS[:4]          # RealBlur holds single images: image methods only
# Mean PSNR of the blurry input over all protocol frames (frame-weighted, as the method scores are). RealBlur: the
# official evaluation (ECC alignment to the sharp image, as for the methods), filled from rb_cells.tsv when present.
INPUT_PSNR = {"gopro": 25.6401, "reds_val": 26.2657, "bsd2": 26.6321, "dvd": 27.2856}
TRAIN_LABEL = {"GoPro": "GoPro", "AiO-mix": "all-in-one mix", "GoPro+DVD+NFS": "GoPro + DVD + NFS",
               "DVD": "DVD", "REDS": "REDS", "BSD1": "BSD 1ms/8ms", "BSD2": "BSD 2ms/16ms", "BSD3": "BSD 3ms/24ms",
               "RealBlurJ": "RealBlur-J", "RealBlurR": "RealBlur-R", "RSBlur": "RSBlur", "HIDE": "HIDE", "NAH": "NAH",
               "REDS_ours": "REDS (retrained by us)", "GoPro_ours": "GoPro (retrained by us)",
               "RealBlur": "RealBlur (subset not stated)", "GoProRealBlurJ": "GoPro + RealBlur-J",
               "REDSjpeg": "REDS with JPEG (NTIRE 2021)", "BSD": "BSD (exposure not stated)"}
# (row, test set) whose test clips were part of the checkpoint's training data: the REDS models of BasicSR (EDVR) and
# VRT 007 were trained on REDS train + val without REDS4, and our REDS test set is the official val split.
SEEN = {("EDVR-REDS", "reds_val"), ("VRT@REDS", "reds_val")}
# training label -> the test set it is in-domain for (the "trained on the test set" filter)
IN_DOMAIN = {"GoPro": "gopro", "REDS": "reds_val", "REDS_ours": "reds_val", "REDSjpeg": "reds_val", "BSD": "bsd2", "BSD1": "bsd2", "BSD2": "bsd2", "BSD3": "bsd2",
             "DVD": "dvd", "RealBlurJ": "realblur_j", "RealBlurR": "realblur_r", "GoPro_ours": "gopro"}
SITE_URL = ""   # the GitHub Pages address once Pages is on, e.g. https://<owner>.github.io/deblurring-is-solved/
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
        "key": r["key"], "name": r["name"], "variant": r["variant"], "task": r["task"],
        "family": r["cluster"], "year": int(r["year"]) if str(r["year"]).isdigit() else None,
        "venue": r["venue"] if r["venue"] not in ("-", "unknown") else "",
        "train": TRAIN_LABEL.get(r["train"], r["train"]), "tr": r["train"], "gopro_trained": r["train"] == "GoPro",
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

data = {
    "datasets": cards, "imgBase": IMG_BASE,
    "sets": SETS, "setLabel": SET_LABEL, "full": FULL, "input": INPUT_PSNR,
    "seq": seq_names, "seqInput": {ds: [round(seq_in[ds][s], 3) if s in seq_in.get(ds, {}) else None
                                        for s in seq_names[ds]] for ds in SETS},
    "rows": rows, "extra": sorted(iqa_have), "ref": iqa_ref, "videoSets": VIDEO_SETS,
    "trains": sorted({(r["tr"], r["train"]) for r in rows}, key=lambda t: (t[0] != "GoPro", t[1])),
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
(out / "index.html").write_text(page, encoding="utf-8")
(out / ".nojekyll").write_text("", encoding="utf-8")

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
       f"> [!NOTE]\n> Numbers as of {FACTS['built']}: {FACTS['n_complete']} of {FACTS['n_rows']} variants scored on all their sets (six for image methods, four for video methods), "
       f"{FACTS['n_some']} on some. GoPro-trained weights (other training sets are on the site), PSNR in dB (RealBlur: official ECC-aligned PSNR), sorted by PSNR. The full table is in "
       f"[`docs/leaderboard.csv`](docs/leaderboard.csv) and on the {site_link}.\n\n"
       + "\n\n".join(blocks) + "\n<!-- generated:end -->")
readme = ROOT / "README.md"
txt = readme.read_text(encoding="utf-8")
a, b = txt.find("<!-- generated:start"), txt.find("<!-- generated:end -->")
if a >= 0 and b > a:
    readme.write_text(txt[:a] + gen + txt[b + len("<!-- generated:end -->"):], encoding="utf-8")
else:
    print("README.md has no generated block; left unchanged")

if FRAGMENT:
    frag = re.sub(r"<!doctype html>\s*|</?html[^>]*>\s*|</?head>\s*|</?body>\s*", "", page, flags=re.I)
    frag = re.sub(r'<meta (charset|name="viewport")[^>]*>\s*', "", frag)
    # the artifact preview serves no binary type: the frame data go there as <file>.b64.txt (tools/stage_preview.py)
    frag = frag.replace("<script>/* Two methods, frame by frame", "<script>window.DBIS_B64 = true;</script>\n<script>/* Two methods, frame by frame", 1)
    Path(FRAGMENT).write_text(frag, encoding="utf-8")

print(f"rows {len(rows)} (complete {len(complete_all)}, partial {len(some)}, none {len(none_yet)}), page {len(page) / 1024:.0f} KiB")
