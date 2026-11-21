/* Two methods, frame by frame (#pairs) and the frame viewer (#frames). Inlined into the page by tools/build_site.py.
   Data, fetched on demand next to the page:
     data/frames_<set>.json   frames with their characteristics, row keys, metric codecs
     data/pf_<set>_<m>.bin    uint16 LE, rows x frames, value = off + q * scale, 65535 = not scored
     img/index.json           frames exported for the viewer, per set
     img/<set>/<n>.bin        b"DBIS", uint32 LE header length, JSON header, then WebP files (images, error maps) */
(function () {
  "use strict";
  var B = window.DBIS;
  if (!B) return;
  var h = B.h, s = B.s, esc = B.esc, minus = B.minus, METRIC = B.metric, LBL = B.label;
  var byKey = {};
  B.rows.forEach(function (r) { byKey[r.key] = r; });
  var CH = {
    psnr_in: ["Blur: PSNR of the blurry input, dB", "Lower means stronger blur."],
    grad_ratio: ["Detail left after blur", "Mean gradient of the input over that of the sharp frame."],
    motion: ["Motion, px per frame", "Median optical flow between consecutive sharp frames (MEMFOF)."],
    motion_p90: ["Fast motion, px per frame", "90th percentile of the optical flow magnitude."],
    shift: ["Global shift, px per frame", "Length of the mean flow vector: the camera-like part of the motion."],
    nonuniform: ["Motion non-uniformity", "(p90 − p10) / (median + 1) of the flow magnitude: moving objects and depth."],
    texture: ["Texture", "Mean gradient magnitude of the sharp frame."],
    brightness: ["Brightness", "Mean luma of the sharp frame, 0 to 255."],
    contrast: ["Contrast", "Standard deviation of the luma of the sharp frame."],
    highlights: ["Clipped highlights, %", "Share of input pixels with a channel at 250 or more."],
    noise: ["Input noise σ", "Immerkaer estimate on the input luma."],
    colorfulness: ["Colorfulness", "Hasler and Süsstrunk measure of the sharp frame."],
    psnr_med: ["Difficulty: median PSNR of all methods, dB", "Median over the GoPro-trained methods scored on every frame of the set."],
    psnr_best: ["Best PSNR of any method, dB", "Maximum over the same methods."],
    idx: ["Position in the sequence", "0 is the first frame."]
  };
  var cache = {};
  function getJSON(u) {
    if (!cache[u]) cache[u] = fetch(u).then(function (r) { if (!r.ok) throw new Error(u + ": " + r.status); return r.json(); });
    return cache[u];
  }
  // binary files; hosts that serve no binary type get the same bytes as base64 text next to them (<file>.b64.txt)
  function getBuf(u) {
    if (!cache[u]) {
      if (window.DBIS_B64 && !/^https?:/.test(u)) {
        cache[u] = fetch(u + ".b64.txt").then(function (r) { if (!r.ok) throw new Error(u + ": " + r.status); return r.text(); }).then(function (t) {
          var bin = atob(t.trim()), n = bin.length, out = new Uint8Array(n);
          for (var i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
          return out.buffer;
        });
      } else {
        cache[u] = fetch(u).then(function (r) { if (!r.ok) throw new Error(u + ": " + r.status); return r.arrayBuffer(); });
      }
    }
    return cache[u];
  }
  function quant(a, q) { var b = a.slice().sort(function (x, y) { return x - y; }); return b[Math.min(b.length - 1, Math.max(0, Math.round(q * (b.length - 1))))]; }
  function niceStep(span, n) { var st0 = span / n, p = Math.pow(10, Math.floor(Math.log10(st0))), m = st0 / p; return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p; }
  function fmt(v, st) { var d = Math.max(0, -Math.floor(Math.log10(st) + 1e-9)); return minus(v.toFixed(d)); }
  function rowName(k) { var r = byKey[k]; return r ? r.name : k; }
  function rowLabel(k) {
    var r = byKey[k]; if (!r) return k;
    return r.name + " · " + r.variant.split(",").join(", ") + (r.gopro_trained ? "" : " · trained on " + r.train);
  }

  /* ================================================================ two methods, frame by frame */
  var PW = { ds: null, a: null, b: null, m: "psnr", x: "psnr_in" };
  var pwA = document.getElementById("pw-a"), pwB = document.getElementById("pw-b"), pwM = document.getElementById("pw-m"),
    pwX = document.getElementById("pw-x");
  var IMGIDX = null;
  getJSON("img/index.json").then(function (j) { IMGIDX = j; renderViewer(); drawPairs(); }).catch(function () { IMGIDX = {}; renderViewer(); });

  function setOptions(sel, items, value) {
    sel.replaceChildren.apply(sel, items.map(function (it) { return h("option", { value: it[0] }, it[1]); }));
    if (value != null) sel.value = value;
  }
  function values(F, buf, meta, key) {
    var n = F.frames.length, ri = F.keys.indexOf(key), out = new Float64Array(n);
    if (ri < 0) { out.fill(NaN); return out; }
    var u = new Uint16Array(buf, ri * n * 2, n);
    for (var i = 0; i < n; i++) out[i] = u[i] === 65535 ? NaN : meta.off + u[i] * meta.scale;
    return out;
  }
  function setupPairs(F) {
    var ds = B.ds();
    var keys = F.keys.filter(function (k) { return byKey[k]; });
    var psnr = function (k) { var c = byKey[k].s[ds]; return c && c.psnr != null ? c.psnr : -1; };
    keys.sort(function (a, b) { return psnr(b) - psnr(a); });
    var items = keys.map(function (k) { return [k, rowLabel(k)]; });
    if (PW.ds !== ds) {
      var gop = keys.filter(function (k) { return byKey[k].gopro_trained && byKey[k].s[ds] && byKey[k].s[ds].ok; });
      PW.a = gop[0] || keys[0]; PW.b = gop[1] || keys[1]; PW.ds = ds;
    }
    setOptions(pwA, items, PW.a); setOptions(pwB, items, PW.b);
    var ms = Object.keys(F.metrics).filter(function (m) { return METRIC[m]; });
    ms.sort(function (a, b) { return Object.keys(METRIC).indexOf(a) - Object.keys(METRIC).indexOf(b); });
    if (ms.indexOf(PW.m) < 0) PW.m = ms[0];
    setOptions(pwM, ms.map(function (m) { return [m, METRIC[m].label]; }), PW.m);
    var cs = F.cols.filter(function (c) { return CH[c]; });
    if (cs.indexOf(PW.x) < 0) PW.x = cs[0];
    setOptions(pwX, cs.map(function (c) { return [c, CH[c][0]]; }), PW.x);
  }
  [pwA, pwB, pwM, pwX].forEach(function (el) {
    el.addEventListener("change", function () { PW.a = pwA.value; PW.b = pwB.value; PW.m = pwM.value; PW.x = pwX.value; drawPairs(); });
  });
  document.getElementById("pw-swap").addEventListener("click", function () { var t = PW.a; PW.a = PW.b; PW.b = t; pwA.value = PW.a; pwB.value = PW.b; drawPairs(); });

  var drawSeq = 0;
  function drawPairs() {
    var ds = B.ds(), my = ++drawSeq;
    document.getElementById("pw-title").textContent = "Two methods, frame by frame on " + LBL[ds];
    getJSON("data/frames_" + ds + ".json").then(function (F) {
      if (my !== drawSeq) return;
      setupPairs(F);
      var meta = F.metrics[PW.m];
      return getBuf("data/pf_" + ds + "_" + PW.m + ".bin").then(function (buf) {
        if (my !== drawSeq) return;
        var a = values(F, buf, meta, PW.a), b = values(F, buf, meta, PW.b), up = METRIC[PW.m].up;
        var xi = F.cols.indexOf(PW.x), pts = [];
        for (var i = 0; i < a.length; i++) {
          if (!isFinite(a[i]) || !isFinite(b[i])) continue;
          var x = F.frames[i][xi];
          pts.push({ i: i, a: a[i], b: b[i], d: up ? a[i] - b[i] : b[i] - a[i], x: x == null ? NaN : +x });
        }
        drawHist(pts);
        drawScatter(F, pts);
        drawRho(F, pts);
      });
    }).catch(function (e) {
      ["pw-hist", "pw-scatter", "pw-rho"].forEach(function (id) { document.getElementById(id).replaceChildren(h("p", { class: "fig-note" }, "Per-frame data is not available here (" + e.message + ").")); });
    });
  }
  function unitOf(m) { return m === "psnr" ? " dB" : ""; }
  function legendAB() {
    return [h("span", { class: "key" }, h("i", { class: "sw pa" }), "A: " + rowName(PW.a)), " ",
      h("span", { class: "key" }, h("i", { class: "sw pb" }), "B: " + rowName(PW.b))];
  }

  function drawHist(pts) {
    var box = document.getElementById("pw-hist"), W = Math.max(280, box.clientWidth), H = 260, m = { l: 44, r: 12, t: 34, b: 34 };
    var mlab = METRIC[PW.m].label;
    document.getElementById("pw-h-title").replaceChildren("How much better A is than B on each frame, " + mlab + unitOf(PW.m));
    if (pts.length < 5) { box.replaceChildren(h("p", { class: "fig-note" }, "These two methods share too few scored frames on this set.")); document.getElementById("pw-h-note").textContent = ""; return; }
    var d = pts.map(function (p) { return p.d; }), absd = d.map(Math.abs);
    var R = quant(absd, 0.98) || Math.max.apply(null, absd) || 1, nb = 20, bw = R / nb;
    var cnt = new Array(2 * nb).fill(0), win = 0, loss = 0, tie = 0;
    d.forEach(function (v) {
      if (v > 0) win++; else if (v < 0) loss++; else tie++;
      var k = v === 0 ? -1 : Math.floor(v / bw) + nb;
      if (v === 0) return;
      cnt[Math.max(0, Math.min(2 * nb - 1, k))]++;
    });
    var ymax = Math.max.apply(null, cnt) || 1;
    var X = function (v) { return m.l + (v + R) / (2 * R) * (W - m.l - m.r); }, Y = function (c) { return H - m.b - c / ymax * (H - m.t - m.b); };
    var root = s("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }), ax = s("g", { class: "ax" });
    var yst = niceStep(ymax, 4);
    for (var t = 0; t <= ymax; t += yst) { ax.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" })); ax.append(s("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, String(t))); }
    var xst = niceStep(2 * R, W < 420 ? 4 : 6);
    for (var v = -Math.floor(R / xst) * xst; v <= R + 1e-12; v += xst) ax.append(s("text", { x: X(v), y: H - m.b + 15, "text-anchor": "middle" }, (v > 0 ? "+" : "") + fmt(v, xst)));
    ax.append(s("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "dom" }));
    ax.append(s("text", { x: m.l, y: m.t - 18, class: "ttl" }, "frames"));
    root.append(ax);
    var g = s("g");
    var bpx = (W - m.l - m.r) / (2 * nb);
    cnt.forEach(function (c, k) {
      if (!c) return;
      var x0 = m.l + k * bpx + 1, hgt = H - m.b - Y(c);
      g.append(s("rect", { x: x0, y: Y(c), width: Math.max(1, bpx - 2), height: hgt, rx: Math.min(2, (bpx - 2) / 2), class: k >= nb ? "bar-a" : "bar-b" }));
    });
    root.append(g);
    root.append(s("line", { x1: X(0), x2: X(0), y1: m.t - 4, y2: H - m.b, class: "zero" }));
    var n = d.length, pc = function (x) { return Math.round(100 * x / n) + " %"; };
    root.append(s("text", { x: X(0) - 8, y: m.t - 4, "text-anchor": "end", class: "dlab" }, "B better on " + pc(loss)));
    root.append(s("text", { x: X(0) + 8, y: m.t - 4, "text-anchor": "start", class: "dlab" }, "A better on " + pc(win)));
    box.replaceChildren(root);
    var mean = d.reduce(function (x, y) { return x + y; }, 0) / n, med = quant(d, 0.5), u = unitOf(PW.m);
    var nd = PW.m === "psnr" ? 3 : 4;
    document.getElementById("pw-h-note").replaceChildren(h("span", { class: "keys" }, legendAB()), h("br"),
      n + " frames scored by both. Mean " + (mean > 0 ? "+" : "") + minus(mean.toFixed(nd)) + u + ", median " + (med > 0 ? "+" : "") + minus(med.toFixed(nd)) + u +
      (tie ? ", " + tie + " ties" : "") + ". Bars beyond ±" + minus(R.toFixed(nd)) + u + " hold the 2 % largest differences.");
  }

  function drawScatter(F, pts) {
    var box = document.getElementById("pw-scatter"), W = Math.max(280, box.clientWidth), H = 300, m = { l: 48, r: 14, t: 24, b: 36 };
    var ch = CH[PW.x] || [PW.x, ""];
    document.getElementById("pw-s-title").textContent = "The same difference against " + ch[0].charAt(0).toLowerCase() + ch[0].slice(1);
    var P = pts.filter(function (p) { return isFinite(p.x); });
    if (P.length < 5) { box.replaceChildren(h("p", { class: "fig-note" }, "This characteristic is not measured on this set.")); document.getElementById("pw-s-note").textContent = ch[1]; return; }
    var xs = P.map(function (p) { return p.x; }), ds = P.map(function (p) { return p.d; });
    var xlo = quant(xs, 0.005), xhi = quant(xs, 0.995); if (xhi <= xlo) { xlo = Math.min.apply(null, xs); xhi = Math.max.apply(null, xs) + 1e-9; }
    var R = quant(ds.map(Math.abs), 0.99) || 1;
    var X = function (v) { return m.l + (Math.max(xlo, Math.min(xhi, v)) - xlo) / (xhi - xlo) * (W - m.l - m.r); };
    var Y = function (v) { return m.t + (R - Math.max(-R, Math.min(R, v))) / (2 * R) * (H - m.t - m.b); };
    var root = s("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }), ax = s("g", { class: "ax" });
    var yst = niceStep(2 * R, 4), xst = niceStep(xhi - xlo, W < 420 ? 4 : 6);
    for (var t = -Math.floor(R / yst) * yst; t <= R + 1e-12; t += yst) { ax.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" })); ax.append(s("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, (t > 0 ? "+" : "") + fmt(t, yst))); }
    for (var v = Math.ceil(xlo / xst) * xst; v <= xhi + 1e-12; v += xst) ax.append(s("text", { x: X(v), y: H - m.b + 15, "text-anchor": "middle" }, fmt(v, xst)));
    ax.append(s("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "dom" }));
    ax.append(s("text", { x: W - m.r, y: H - 4, "text-anchor": "end", class: "ttl" }, ch[0]));
    ax.append(s("text", { x: m.l, y: m.t - 10, class: "ttl" }, "A − B, " + METRIC[PW.m].label + unitOf(PW.m) + (METRIC[PW.m].up ? "" : " (sign flipped: up means A better)")));
    root.append(ax);
    root.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "zero" }));
    var withImg = {}, ent = (IMGIDX && IMGIDX[B.ds()]) ? IMGIDX[B.ds()].frames : [];
    ent.forEach(function (e) { withImg[e.seq + "/" + e.stem] = true; });
    var g = s("g"), seqi = F.cols.indexOf("seq"), stemi = F.cols.indexOf("stem");
    P.forEach(function (p) { g.append(s("circle", { cx: X(p.x).toFixed(1), cy: Y(p.d).toFixed(1), r: 2.2, class: p.d >= 0 ? "dot-a" : "dot-b" })); });
    root.append(g);
    var rings = s("g");
    P.forEach(function (p) { var f = F.frames[p.i]; if (withImg[f[seqi] + "/" + f[stemi]]) rings.append(s("circle", { cx: X(p.x), cy: Y(p.d), r: 5.5, class: "ring" })); });
    root.append(rings);
    // binned mean: ten equal-count bins along x
    var srt = P.slice().sort(function (a, b) { return a.x - b.x; }), nbin = Math.min(10, Math.floor(srt.length / 20)) || 1, line = "", bins = [];
    for (var k = 0; k < nbin; k++) {
      var part = srt.slice(Math.floor(k * srt.length / nbin), Math.floor((k + 1) * srt.length / nbin));
      var mx = part.reduce(function (a, p) { return a + p.x; }, 0) / part.length, md = part.reduce(function (a, p) { return a + p.d; }, 0) / part.length;
      bins.push([mx, md]); line += (k ? "L" : "M") + X(mx).toFixed(1) + " " + Y(md).toFixed(1);
    }
    root.append(s("path", { d: line, class: "binline" }));
    bins.forEach(function (b) { root.append(s("circle", { cx: X(b[0]), cy: Y(b[1]), r: 3.5, class: "binpt" })); });
    var hover = s("circle", { r: 6, class: "hov", cx: 0, cy: 0, visibility: "hidden" });
    root.append(hover);
    var overlay = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "hit" });
    root.append(overlay);
    var tip = document.getElementById("tip");
    function nearest(ev) {
      var r = root.getBoundingClientRect(), sx = (ev.clientX - r.left) * W / r.width, sy = (ev.clientY - r.top) * H / r.height, best = null, bd = 144;
      P.forEach(function (p) { var dx = X(p.x) - sx, dy = Y(p.d) - sy, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = p; } });
      return best;
    }
    overlay.addEventListener("pointermove", function (ev) {
      var p = nearest(ev);
      if (!p) { tip.hidden = true; hover.setAttribute("visibility", "hidden"); return; }
      hover.setAttribute("cx", X(p.x)); hover.setAttribute("cy", Y(p.d)); hover.setAttribute("visibility", "visible");
      var f = F.frames[p.i], nd = PW.m === "psnr" ? 2 : 3, has = withImg[f[seqi] + "/" + f[stemi]];
      tip.innerHTML = "<b>" + esc(f[seqi]) + " · " + esc(f[stemi]) + "</b><br><span class='k'>" + esc(ch[0]) + "</span> " + minus((+p.x).toFixed(2)) +
        "<table><tr><td class='k'>A</td><td>" + minus(p.a.toFixed(nd)) + "</td><td class='k'>B</td><td>" + minus(p.b.toFixed(nd)) + "</td></tr></table>" +
        "<span class='k'>" + (has ? "Click to open this frame in the viewer" : "No images exported for this frame") + "</span>";
      tip.hidden = false;
      var pad = 14, w = tip.offsetWidth, hh = tip.offsetHeight, x = ev.clientX + pad, y = ev.clientY + pad;
      if (x + w > innerWidth - 8) x = ev.clientX - w - pad; if (y + hh > innerHeight - 8) y = ev.clientY - hh - pad;
      tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
    });
    overlay.addEventListener("pointerleave", function () { tip.hidden = true; hover.setAttribute("visibility", "hidden"); });
    overlay.addEventListener("click", function (ev) {
      var p = nearest(ev); if (!p) return;
      var f = F.frames[p.i];
      if (withImg[f[seqi] + "/" + f[stemi]]) openFrame(f[seqi], f[stemi], PW.a, PW.b);
    });
    box.replaceChildren(root);
    document.getElementById("pw-s-note").replaceChildren(ch[1] + " Each dot is one frame; the line joins the mean difference of ten equal-size groups of frames along the axis. Ringed frames can be opened in the viewer below.");
  }

  function ranks(v) {
    var idx = v.map(function (x, i) { return i; }).sort(function (a, b) { return v[a] - v[b]; }), r = new Array(v.length);
    for (var i = 0; i < idx.length;) { var j = i; while (j + 1 < idx.length && v[idx[j + 1]] === v[idx[i]]) j++; for (var k = i; k <= j; k++) r[idx[k]] = (i + j) / 2; i = j + 1; }
    return r;
  }
  function pearson(a, b) {
    var n = a.length, ma = 0, mb = 0; for (var i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
    var sab = 0, saa = 0, sbb = 0; for (i = 0; i < n; i++) { var x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
  }
  function drawRho(F, pts) {
    var box = document.getElementById("pw-rho"), W = Math.max(280, box.clientWidth);
    var cs = F.cols.filter(function (c) { return CH[c]; }), res = [];
    cs.forEach(function (c) {
      var ci = F.cols.indexOf(c), xs = [], ds = [];
      pts.forEach(function (p) { var x = F.frames[p.i][ci]; if (x != null && isFinite(x)) { xs.push(+x); ds.push(p.d); } });
      if (xs.length > 20) res.push([c, pearson(ranks(xs), ranks(ds)), xs.length]);
    });
    res.sort(function (a, b) { return Math.abs(b[1]) - Math.abs(a[1]); });
    var rowH = 24, H = res.length * rowH + 28, lw = Math.min(260, W * 0.45), m = { l: lw, r: 50 }, mid = m.l + (W - m.l - m.r) / 2, half = (W - m.l - m.r) / 2;
    var root = s("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H });
    var ax = s("g", { class: "ax" });
    [-1, -0.5, 0, 0.5, 1].forEach(function (t) { ax.append(s("line", { x1: mid + t * half, x2: mid + t * half, y1: 4, y2: H - 22, class: t ? "grid" : "dom" })); ax.append(s("text", { x: mid + t * half, y: H - 6, "text-anchor": "middle" }, minus(String(t)))); });
    root.append(ax);
    res.forEach(function (r, i) {
      var y = 6 + i * rowH, x0 = Math.min(mid, mid + r[1] * half), w = Math.abs(r[1]) * half;
      var g = s("g", { class: "rhorow" + (r[0] === PW.x ? " on" : "") });
      g.append(s("text", { x: m.l - 10, y: y + 13, "text-anchor": "end", class: "rholab" }, CH[r[0]][0]));
      g.append(s("rect", { x: x0, y: y + 3, width: Math.max(1, w), height: rowH - 9, rx: 2, class: "rhobar" }));
      g.append(s("text", { x: r[1] >= 0 ? mid + w + 6 : mid - w - 6, y: y + 13, "text-anchor": r[1] >= 0 ? "start" : "end", class: "rhoval" }, minus(r[1].toFixed(2))));
      var hit = s("rect", { x: 0, y: y, width: W, height: rowH, class: "hit" });
      hit.addEventListener("click", function () { PW.x = r[0]; pwX.value = r[0]; drawPairs(); });
      g.append(hit);
      root.append(g);
    });
    box.replaceChildren(root);
    document.getElementById("pw-r-note").textContent = "Spearman rank correlation between each characteristic and the per-frame difference A − B (" + pts.length +
      " frames). Positive: A gains more over B as the characteristic grows. Click a row to plot it above.";
  }

  /* ================================================================ frame viewer */
  var FV = { ds: null, n: -1, sub: "all", left: "input", right: null, mode: "slider", err: false, gain: 4, zoom: 4, split: 0.5, cx: 0.5, cy: 0.5, imgs: null, want: null };
  var stage = document.getElementById("fv-stage"), strip = document.getElementById("fv-strip"), fvL = document.getElementById("fv-l"), fvR = document.getElementById("fv-r");
  function segInit(id, key, onchange) {
    var el = document.getElementById(id);
    el.querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        el.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        var v = b.getAttribute("data-v"); FV[key] = isNaN(+v) ? (v === "true" ? true : v === "false" ? false : v) : +v; onchange();
      });
    });
  }
  segInit("fv-mode", "mode", function () { drawStage(); });
  segInit("fv-err", "err", function () { document.getElementById("fv-gain").hidden = !FV.err; drawStage(); });
  segInit("fv-gain", "gain", function () { drawStage(); });
  segInit("fv-zoom", "zoom", function () { drawStage(); });
  fvL.addEventListener("change", function () { FV.left = fvL.value; drawStage(); });
  fvR.addEventListener("change", function () { FV.right = fvR.value; drawStage(); });

  function framesOf(ds) { return IMGIDX && IMGIDX[ds] ? IMGIDX[ds].frames : []; }
  function renderViewer() {
    var ds = B.ds();
    document.getElementById("fv-title").textContent = "Frames of " + LBL[ds];
    var fr = framesOf(ds), subs = ["all"];
    fr.forEach(function (e) { e.why.forEach(function (w) { if (subs.indexOf(w) < 0) subs.push(w); }); });
    if (FV.ds !== ds) { FV.ds = ds; FV.n = -1; FV.sub = "all"; FV.imgs = null; }
    var segEl = document.getElementById("fv-subset");
    segEl.replaceChildren.apply(segEl, subs.map(function (w) {
      var b = h("button", { type: "button", "aria-pressed": w === FV.sub ? "true" : "false" }, w === "all" ? "All frames" : w.charAt(0).toUpperCase() + w.slice(1));
      b.addEventListener("click", function () { FV.sub = w; renderViewer(); });
      return b;
    }));
    var shown = fr.map(function (e, i) { return [e, i]; }).filter(function (p) { return FV.sub === "all" || p[0].why.indexOf(FV.sub) >= 0; });
    strip.replaceChildren.apply(strip, shown.map(function (p) {
      var e = p[0], i = p[1];
      var card = h("button", { type: "button", class: "fv-card", "aria-pressed": i === FV.n ? "true" : "false" },
        e.thumb ? h("img", { src: e.thumb, alt: "", loading: "lazy" }) : null,
        h("span", { class: "fv-why" }, e.why.join(", ")), h("span", { class: "fv-id" }, e.seq + " · " + e.stem),
        h("span", { class: "fv-id" }, "input " + (e.psnr && e.psnr.input != null ? e.psnr.input.toFixed(2) + " dB" : "")));
      card.addEventListener("click", function () { loadFrame(i); });
      return card;
    }));
    if (!fr.length) {
      stage.replaceChildren(h("p", { class: "fig-note" }, "No frames of this set are exported for viewing yet."));
      document.getElementById("fv-note").textContent = "";
      return;
    }
    if (FV.n < 0) loadFrame(shown.length ? shown[0][1] : 0);
  }

  function loadFrame(i, want) {
    var ds = B.ds(), e = framesOf(ds)[i];
    if (!e) return;
    FV.n = i; FV.want = want || null;
    strip.querySelectorAll(".fv-card").forEach(function (c, k) { c.setAttribute("aria-pressed", "false"); });
    renderStripPressed();
    stage.replaceChildren(h("p", { class: "fig-note" }, "Loading " + e.seq + " · " + e.stem + " (" + e.mb + " MB)…"));
    getBuf((B.imgBase || "") + e.file).then(function (buf) {
      var dv = new DataView(buf), hl = dv.getUint32(4, true);
      var head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl))), base = 8 + hl, jobs = [], imgs = { head: head, img: {}, err: {} };
      head.items.forEach(function (it) {
        var url = URL.createObjectURL(new Blob([new Uint8Array(buf, base + it.off, it.len)], { type: "image/webp" }));
        var im = new Image(); im.src = url;
        jobs.push(im.decode().then(function () { imgs[it.kind][it.key] = im; }));
      });
      return Promise.all(jobs).then(function () {
        if (FV.n !== i || B.ds() !== ds) return;
        FV.imgs = imgs; FV.errCache = {};
        var keys = Object.keys(imgs.img).filter(function (k) { return k !== "sharp" && k !== "input"; });
        var opts = [["sharp", "Sharp frame"], ["input", "Blurry input" + ps(e, "input")]].concat(keys.map(function (k) { return [k, rowName(k) + ps(e, k)]; }));
        var w = FV.want;
        FV.left = w && imgs.img[w[0]] ? w[0] : (imgs.img[FV.left] ? FV.left : "input");
        FV.right = w && imgs.img[w[1]] ? w[1] : (FV.right && imgs.img[FV.right] && FV.right !== FV.left ? FV.right : keys[0]);
        setOptions(fvL, opts, FV.left); setOptions(fvR, opts, FV.right);
        drawStage();
      });
    }).catch(function (err) { stage.replaceChildren(h("p", { class: "fig-note" }, "Could not load this frame: " + err.message)); });
  }
  function renderStripPressed() {
    var fr = framesOf(B.ds()), shown = fr.map(function (e, i) { return i; }).filter(function (i) { return FV.sub === "all" || fr[i].why.indexOf(FV.sub) >= 0; });
    strip.querySelectorAll(".fv-card").forEach(function (c, k) { c.setAttribute("aria-pressed", shown[k] === FV.n ? "true" : "false"); });
  }
  function ps(e, k) { return e.psnr && e.psnr[k] != null ? " · " + e.psnr[k].toFixed(2) + " dB" : ""; }
  function openFrame(seq, stem, a, b) {
    var fr = framesOf(B.ds());
    for (var i = 0; i < fr.length; i++) if (fr[i].seq === seq && fr[i].stem === stem) {
      FV.sub = "all"; renderViewer(); loadFrame(i, [a, b]);
      document.getElementById("frames").scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
  }

  // error map: |X - sharp| (grey, from the lossless outputs) times the gain, clipped
  function errSource(k) {
    var key = k + "@" + FV.gain;
    if (FV.errCache[key]) return FV.errCache[key];
    var im = FV.imgs.err[k], cv = document.createElement("canvas");
    cv.width = im.naturalWidth; cv.height = im.naturalHeight;
    var cx = cv.getContext("2d", { willReadFrequently: true });
    cx.drawImage(im, 0, 0);
    var d = cx.getImageData(0, 0, cv.width, cv.height), a = d.data, gn = FV.gain;
    for (var p = 0; p < a.length; p += 4) { var v = Math.min(255, a[p] * gn); a[p] = a[p + 1] = a[p + 2] = v; }
    cx.putImageData(d, 0, 0);
    FV.errCache[key] = cv;
    return cv;
  }
  function source(k) {
    if (!FV.err || k === "sharp" || !FV.imgs.err[k]) return FV.imgs.img[k];
    return errSource(k);
  }
  function label(k) {
    var e = framesOf(B.ds())[FV.n];
    var nm = k === "sharp" ? "Sharp frame" : k === "input" ? "Blurry input" : rowName(k);
    return nm + (k !== "sharp" ? ps(e, k) : "") + (FV.err && k !== "sharp" ? " · error ×" + FV.gain : "");
  }
  function fitCanvas(cv, w, hgt) {
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); cv.style.width = w + "px"; cv.style.height = hgt + "px";
    var c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.imageSmoothingQuality = "high"; return c;
  }
  function drawStage() {
    if (!FV.imgs) return;
    var hd = FV.imgs.head, iw = hd.w, ih = hd.h, W = stage.clientWidth, maxH = Math.max(420, window.innerHeight * 0.78);
    document.getElementById("fv-zoom").hidden = FV.mode !== "zoom";
    var L = FV.left, R = FV.right;
    if (FV.mode === "slider") {
      var w = Math.min(W, maxH * iw / ih), hh = w * ih / iw;
      var pane = h("div", { class: "fv-pane", style: "width:" + w + "px" }), cv = h("canvas", { tabindex: "0", "aria-label": "Divider between " + label(L) + " and " + label(R) + "; arrow keys move it" });
      var c = fitCanvas(cv, w, hh);
      var paint = function () {
        var sx = FV.split * w;
        c.clearRect(0, 0, w, hh);
        c.drawImage(source(L), 0, 0, w, hh);
        c.save(); c.beginPath(); c.rect(sx, 0, w - sx, hh); c.clip(); c.drawImage(source(R), 0, 0, w, hh); c.restore();
        c.fillStyle = "rgba(0,0,0,.35)"; c.fillRect(sx - 2, 0, 4, hh);
        c.fillStyle = "#ffffff"; c.fillRect(sx - 1, 0, 2, hh);
        c.beginPath(); c.arc(sx, hh / 2, 11, 0, 2 * Math.PI); c.fill(); c.strokeStyle = "rgba(0,0,0,.45)"; c.lineWidth = 1; c.stroke();
        c.fillStyle = "#0d0e11"; c.beginPath(); c.moveTo(sx - 3, hh / 2 - 4); c.lineTo(sx - 7, hh / 2); c.lineTo(sx - 3, hh / 2 + 4); c.fill();
        c.beginPath(); c.moveTo(sx + 3, hh / 2 - 4); c.lineTo(sx + 7, hh / 2); c.lineTo(sx + 3, hh / 2 + 4); c.fill();
      };
      paint();
      var drag = false, move = function (ev) { var r = cv.getBoundingClientRect(); FV.split = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)); paint(); };
      cv.addEventListener("pointerdown", function (ev) { drag = true; cv.setPointerCapture(ev.pointerId); move(ev); });
      cv.addEventListener("pointermove", function (ev) { if (drag) move(ev); });
      cv.addEventListener("pointerup", function () { drag = false; });
      cv.addEventListener("keydown", function (ev) { if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") { ev.preventDefault(); FV.split = Math.max(0, Math.min(1, FV.split + (ev.key === "ArrowLeft" ? -0.02 : 0.02))); paint(); } });
      pane.append(cv, h("span", { class: "fv-lab", style: "left:.4rem" }, label(L)), h("span", { class: "fv-lab", style: "right:.4rem" }, label(R)));
      stage.replaceChildren(pane);
    } else {
      var two = W >= 640, pw = two ? (W - 8) / 2 : W, ph = Math.min(pw * ih / iw, maxH / (two ? 1 : 2)), pw2 = ph * iw / ih;
      var mk = function (k) {
        var cv = h("canvas"), c = fitCanvas(cv, pw2, ph);
        return { k: k, cv: cv, c: c, pane: h("div", { class: "fv-pane" }, cv, h("span", { class: "fv-lab", style: "left:.4rem" }, label(k))) };
      };
      var A = mk(L), Bp = mk(R), grid = h("div", { class: "fv-two" }, A.pane, Bp.pane), kids = [grid];
      var zw = FV.zoom, zbox = null, ZA = null, ZB = null;
      if (FV.mode === "zoom") {
        var zpw = two ? (W - 8) / 2 : W, zph = Math.min(zpw * 0.62, 360);
        var mkz = function (k) { var cv = h("canvas"), c = fitCanvas(cv, zpw, zph); c.imageSmoothingEnabled = false; return { k: k, cv: cv, c: c, pane: h("div", { class: "fv-pane" }, cv, h("span", { class: "fv-lab", style: "left:.4rem" }, label(k) + " · zoom " + zw + "×")) }; };
        ZA = mkz(L); ZB = mkz(R); zbox = { w: zpw, h: zph };
        kids.push(h("div", { class: "fv-two" }, ZA.pane, ZB.pane));
      }
      var paint2 = function () {
        [A, Bp].forEach(function (p) {
          p.c.clearRect(0, 0, pw2, ph); p.c.drawImage(source(p.k), 0, 0, pw2, ph);
          if (zbox) {
            var rw = zbox.w / zw, rh = zbox.h / zw;                 // region in image pixels
            var x0 = Math.max(0, Math.min(iw - rw, FV.cx * iw - rw / 2)), y0 = Math.max(0, Math.min(ih - rh, FV.cy * ih - rh / 2)), sc = pw2 / iw;
            p.c.strokeStyle = "#ffffff"; p.c.lineWidth = 2; p.c.strokeRect(x0 * sc, y0 * sc, rw * sc, rh * sc);
            p.c.strokeStyle = "rgba(0,0,0,.5)"; p.c.lineWidth = 1; p.c.strokeRect(x0 * sc - 1.5, y0 * sc - 1.5, rw * sc + 3, rh * sc + 3);
            p.reg = [x0, y0, rw, rh];
          }
        });
        if (zbox) [[ZA, A], [ZB, Bp]].forEach(function (q) {
          var r = q[1].reg; q[0].c.imageSmoothingEnabled = false; q[0].c.clearRect(0, 0, zbox.w, zbox.h);
          q[0].c.drawImage(source(q[0].k), r[0], r[1], r[2], r[3], 0, 0, zbox.w, zbox.h);
        });
      };
      paint2();
      if (zbox) [A, Bp].forEach(function (p) {
        var drag = false, mv = function (ev) { var r = p.cv.getBoundingClientRect(); FV.cx = (ev.clientX - r.left) / r.width; FV.cy = (ev.clientY - r.top) / r.height; paint2(); };
        p.cv.style.cursor = "crosshair"; p.cv.style.touchAction = "none";
        p.cv.addEventListener("pointerdown", function (ev) { drag = true; p.cv.setPointerCapture(ev.pointerId); mv(ev); });
        p.cv.addEventListener("pointermove", function (ev) { if (drag) mv(ev); });
        p.cv.addEventListener("pointerup", function () { drag = false; });
      });
      stage.replaceChildren.apply(stage, kids);
    }
    var e = framesOf(B.ds())[FV.n];
    document.getElementById("fv-note").textContent = e.seq + " · " + e.stem + ", " + hd.w + "×" + hd.h + ". Picked as: " + e.why.join(", ") + ". " +
      (FV.mode === "slider" ? "Drag the white line, or focus the image and use the arrow keys. " : FV.mode === "zoom" ? "Drag on either image to move the zoom window; both sides follow. " : "") +
      (FV.err ? "Error map: |output − sharp| averaged over RGB from the lossless outputs, multiplied by " + FV.gain + ", black is exact. " : "") +
      "Images are WebP (quality 85) of the stored lossless outputs; the sharp frame is lossless.";
  }

  /* ================================================================ one method across the frames of a set */
  var OM = { ds: null, r: null, m: "psnr", x: "psnr_in", v: "score" };
  var omR = document.getElementById("om-r"), omM = document.getElementById("om-m"), omX = document.getElementById("om-x");
  [omR, omM, omX].forEach(function (el) { el.addEventListener("change", function () { OM.r = omR.value; OM.m = omM.value; OM.x = omX.value; drawOne(); }); });
  (function () {
    var el = document.getElementById("om-v");
    el.querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        el.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        OM.v = b.getAttribute("data-v"); drawOne();
      });
    });
  })();
  var NBIN = 5, BINLAB = ["lowest fifth", "second", "middle", "fourth", "highest fifth"];
  function median(a) { var b = a.slice().sort(function (x, y) { return x - y; }), n = b.length; return n ? (n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2) : NaN; }

  var oneSeq = 0;
  function drawOne() {
    var ds = B.ds(), my = ++oneSeq;
    document.getElementById("om-title").textContent = "One method across the frames of " + LBL[ds];
    getJSON("data/frames_" + ds + ".json").then(function (F) {
      if (my !== oneSeq) return;
      var keys = F.keys.filter(function (k) { return byKey[k]; });
      var psnr = function (k) { var c = byKey[k].s[ds]; return c && c.psnr != null ? c.psnr : -1; };
      keys.sort(function (a, b) { return psnr(b) - psnr(a); });
      if (OM.ds !== ds || keys.indexOf(OM.r) < 0) {
        var gop = keys.filter(function (k) { return byKey[k].gopro_trained && byKey[k].s[ds] && byKey[k].s[ds].ok; });
        OM.r = gop[0] || keys[0]; OM.ds = ds;
      }
      setOptions(omR, keys.map(function (k) { return [k, rowLabel(k)]; }), OM.r);
      var ms = Object.keys(F.metrics).filter(function (m) { return METRIC[m]; });
      ms.sort(function (a, b) { return Object.keys(METRIC).indexOf(a) - Object.keys(METRIC).indexOf(b); });
      if (ms.indexOf(OM.m) < 0) OM.m = ms[0];
      setOptions(omM, ms.map(function (m) { return [m, METRIC[m].label]; }), OM.m);
      var cs = F.cols.filter(function (c) { return CH[c]; });
      if (cs.indexOf(OM.x) < 0) OM.x = cs[0];
      setOptions(omX, cs.map(function (c) { return [c, CH[c][0]]; }), OM.x);
      var gainBtn = document.querySelector('#om-v button[data-v="gain"]');
      gainBtn.disabled = OM.m !== "psnr";
      if (OM.v === "gain" && OM.m !== "psnr") { OM.v = "score"; document.querySelectorAll("#om-v button").forEach(function (x) { x.setAttribute("aria-pressed", x.getAttribute("data-v") === "score" ? "true" : "false"); }); }
      return getBuf("data/pf_" + ds + "_" + OM.m + ".bin").then(function (buf) {
        if (my !== oneSeq) return;
        var meta = F.metrics[OM.m], n = F.frames.length, up = METRIC[OM.m].up;
        // the field: GoPro-trained rows scored on at least 90 % of the frames; the chosen row is ranked among them
        var pool = keys.filter(function (k) { return byKey[k].gopro_trained || k === OM.r; }), V = {};
        pool.forEach(function (k) { V[k] = values(F, buf, meta, k); });
        pool = pool.filter(function (k) { var c = 0; for (var i = 0; i < n; i++) if (isFinite(V[k][i])) c++; return c >= 0.9 * n || k === OM.r; });
        var med = new Float64Array(n);
        for (var i = 0; i < n; i++) {
          var col = [];
          pool.forEach(function (k) { if (k !== OM.r || byKey[k].gopro_trained) { var v = V[k][i]; if (isFinite(v)) col.push(v); } });
          med[i] = col.length ? median(col) : NaN;
        }
        var pin = F.cols.indexOf("psnr_in");
        var yOf = function (k, i) {
          var v = V[k][i];
          if (OM.v === "med") return up ? v - med[i] : med[i] - v;
          if (OM.v === "gain") return v - F.frames[i][pin];
          return v;
        };
        drawOneScatter(F, V, med, yOf, pool);
        drawProfile(F, V, med, pool);
      });
    }).catch(function (e) {
      ["om-scatter", "om-prof"].forEach(function (id) { document.getElementById(id).replaceChildren(h("p", { class: "fig-note" }, "Per-frame data is not available here (" + e.message + ").")); });
    });
  }

  function binsOf(F, c) {
    var ci = F.cols.indexOf(c), xs = [];
    F.frames.forEach(function (f, i) { var x = f[ci]; if (x != null && isFinite(x)) xs.push([+x, i]); });
    xs.sort(function (a, b) { return a[0] - b[0]; });
    var bin = new Int8Array(F.frames.length).fill(-1), edges = [];
    xs.forEach(function (p, j) { bin[p[1]] = Math.min(NBIN - 1, Math.floor(j * NBIN / xs.length)); });
    for (var b = 0; b <= NBIN; b++) edges.push(xs.length ? xs[Math.min(xs.length - 1, Math.floor(b * xs.length / NBIN))][0] : NaN);
    return { bin: bin, edges: edges, n: xs.length };
  }

  function drawOneScatter(F, V, med, yOf, pool) {
    var box = document.getElementById("om-scatter"), W = Math.max(280, box.clientWidth), H = W > 700 ? 340 : 300, m = { l: 50, r: 14, t: 24, b: 36 };
    var r = byKey[OM.r], ch = CH[OM.x] || [OM.x, ""], ci = F.cols.indexOf(OM.x), up = METRIC[OM.m].up;
    var ylab = OM.v === "med" ? METRIC[OM.m].label + " minus the median method" + (up ? "" : " (sign flipped: up means better)") : OM.v === "gain" ? "PSNR gain over the blurry input, dB" : METRIC[OM.m].label + (OM.m === "psnr" ? ", dB" : "");
    var P = [];
    F.frames.forEach(function (f, i) { var x = f[ci], y = yOf(OM.r, i); if (x != null && isFinite(x) && isFinite(y)) P.push({ i: i, x: +x, y: y }); });
    if (P.length < 10) { box.replaceChildren(h("p", { class: "fig-note" }, ch[0] + " is not measured on this set, or this method has no per-frame scores here.")); document.getElementById("om-s-note").textContent = ""; document.getElementById("om-s-title").textContent = ""; return; }
    var rho = pearson(ranks(P.map(function (p) { return p.x; })), ranks(P.map(function (p) { return p.y; })));
    document.getElementById("om-s-title").textContent = rowName(OM.r) + ": " + (OM.v === "score" ? METRIC[OM.m].label : OM.v === "med" ? "difference to the median method" : "gain over the input") + " against " + ch[0].charAt(0).toLowerCase() + ch[0].slice(1) + ", Spearman ρ = " + minus(rho.toFixed(2));
    var xs = P.map(function (p) { return p.x; }), ys = P.map(function (p) { return p.y; });
    var xlo = quant(xs, 0.005), xhi = quant(xs, 0.995); if (xhi <= xlo) { xlo = Math.min.apply(null, xs); xhi = Math.max.apply(null, xs) + 1e-9; }
    var ylo = quant(ys, 0.01), yhi = quant(ys, 0.99);
    if (OM.v === "score") { var mv = P.map(function (p) { return med[p.i]; }).filter(isFinite); if (mv.length) { ylo = Math.min(ylo, quant(mv, 0.05)); yhi = Math.max(yhi, quant(mv, 0.95)); } }
    var pad = (yhi - ylo) * 0.06 || 0.1; ylo -= pad; yhi += pad;
    if (OM.v !== "score") { var R = Math.max(Math.abs(ylo), Math.abs(yhi)); if (OM.v === "med") { ylo = -R; yhi = R; } }
    var X = function (v) { return m.l + (Math.max(xlo, Math.min(xhi, v)) - xlo) / (xhi - xlo) * (W - m.l - m.r); };
    var Y = function (v) { return m.t + (yhi - Math.max(ylo, Math.min(yhi, v))) / (yhi - ylo) * (H - m.t - m.b); };
    var root = s("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H }), ax = s("g", { class: "ax" });
    var yst = niceStep(yhi - ylo, 5), xst = niceStep(xhi - xlo, W < 420 ? 4 : 6);
    for (var t = Math.ceil(ylo / yst) * yst; t <= yhi + 1e-12; t += yst) { ax.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" })); ax.append(s("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, (OM.v !== "score" && t > 0 ? "+" : "") + fmt(t, yst))); }
    for (var v = Math.ceil(xlo / xst) * xst; v <= xhi + 1e-12; v += xst) ax.append(s("text", { x: X(v), y: H - m.b + 15, "text-anchor": "middle" }, fmt(v, xst)));
    ax.append(s("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "dom" }));
    ax.append(s("text", { x: W - m.r, y: H - 4, "text-anchor": "end", class: "ttl" }, ch[0]));
    ax.append(s("text", { x: m.l, y: m.t - 10, class: "ttl" }, ylab));
    root.append(ax);
    if (OM.v !== "score") root.append(s("line", { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), class: "zero" }));
    var g = s("g");
    P.forEach(function (p) { g.append(s("circle", { cx: X(p.x).toFixed(1), cy: Y(p.y).toFixed(1), r: 2.2, class: "om-dot " + (r ? r.task : "image") })); });
    root.append(g);
    // fifths along the characteristic: mean of this method, and (score view) of the median method
    var B5 = binsOf(F, OM.x), mx = [], my = [], mm = [];
    for (var b = 0; b < NBIN; b++) {
      var px = [], py = [], pm = [];
      P.forEach(function (p) { if (B5.bin[p.i] === b) { px.push(p.x); py.push(p.y); if (isFinite(med[p.i])) pm.push(med[p.i]); } });
      if (!px.length) continue;
      var avg = function (a) { return a.reduce(function (u, w) { return u + w; }, 0) / a.length; };
      mx.push(avg(px)); my.push(avg(py)); mm.push(pm.length ? avg(pm) : NaN);
    }
    if (OM.v === "score") {
      var dm = ""; mx.forEach(function (x, k) { if (isFinite(mm[k])) dm += (dm ? "L" : "M") + X(x).toFixed(1) + " " + Y(mm[k]).toFixed(1); });
      root.append(s("path", { d: dm, class: "medline" }));
    }
    var d = ""; mx.forEach(function (x, k) { d += (k ? "L" : "M") + X(x).toFixed(1) + " " + Y(my[k]).toFixed(1); });
    root.append(s("path", { d: d, class: "binline" }));
    mx.forEach(function (x, k) { root.append(s("circle", { cx: X(x), cy: Y(my[k]), r: 3.5, class: "binpt" })); });
    var hover = s("circle", { r: 6, class: "hov", cx: 0, cy: 0, visibility: "hidden" });
    root.append(hover);
    var overlay = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "hit" });
    root.append(overlay);
    var tip = document.getElementById("tip"), seqi = F.cols.indexOf("seq"), stemi = F.cols.indexOf("stem");
    var withImg = {};
    ((IMGIDX && IMGIDX[B.ds()]) ? IMGIDX[B.ds()].frames : []).forEach(function (e) { withImg[e.seq + "/" + e.stem] = true; });
    function nearest(ev) {
      var rr = root.getBoundingClientRect(), sx = (ev.clientX - rr.left) * W / rr.width, sy = (ev.clientY - rr.top) * H / rr.height, best = null, bd = 144;
      P.forEach(function (p) { var dx = X(p.x) - sx, dy = Y(p.y) - sy, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = p; } });
      return best;
    }
    overlay.addEventListener("pointermove", function (ev) {
      var p = nearest(ev);
      if (!p) { tip.hidden = true; hover.setAttribute("visibility", "hidden"); return; }
      hover.setAttribute("cx", X(p.x)); hover.setAttribute("cy", Y(p.y)); hover.setAttribute("visibility", "visible");
      var f = F.frames[p.i], has = withImg[f[seqi] + "/" + f[stemi]];
      tip.innerHTML = "<b>" + esc(f[seqi]) + " · " + esc(f[stemi]) + "</b><br><span class='k'>" + esc(ch[0]) + "</span> " + minus((+p.x).toFixed(2)) +
        "<br><span class='k'>" + esc(ylab) + "</span> " + minus(p.y.toFixed(OM.m === "psnr" ? 2 : 3)) +
        (has ? "<br><span class='k'>Click to open this frame in the viewer</span>" : "");
      tip.hidden = false;
      var pad = 14, w = tip.offsetWidth, hh = tip.offsetHeight, x = ev.clientX + pad, y = ev.clientY + pad;
      if (x + w > innerWidth - 8) x = ev.clientX - w - pad; if (y + hh > innerHeight - 8) y = ev.clientY - hh - pad;
      tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
    });
    overlay.addEventListener("pointerleave", function () { tip.hidden = true; hover.setAttribute("visibility", "hidden"); });
    overlay.addEventListener("click", function (ev) { var p = nearest(ev); if (!p) return; var f = F.frames[p.i]; if (withImg[f[seqi] + "/" + f[stemi]]) openFrame(f[seqi], f[stemi], "input", OM.r); });
    box.replaceChildren(root);
    document.getElementById("om-s-note").replaceChildren(h("span", { class: "keys" },
      h("span", { class: "key" }, h("i", { class: "sw " + (r ? r.task : "image") }), "one frame"),
      h("span", { class: "key" }, h("i", { class: "sw front" }), "mean over each fifth of the frames"),
      OM.v === "score" ? h("span", { class: "key" }, h("i", { class: "sw medk" }), "the same for the median method") : null),
      h("br"), ch[1] + " The median method is the per-frame median over the " + pool.filter(function (k) { return byKey[k].gopro_trained; }).length + " GoPro-trained methods scored on this set.");
  }

  function drawProfile(F, V, med, pool) {
    var box = document.getElementById("om-prof"), up = METRIC[OM.m].up, n = F.frames.length;
    var cs = F.cols.filter(function (c) { return CH[c] && c !== "psnr_best"; });
    var field = pool.filter(function (k) { return byKey[k].gopro_trained || k === OM.r; });
    var head = h("tr", null, h("th", { scope: "col" }, "Characteristic"), BINLAB.map(function (b, k) { return h("th", { scope: "col", class: "n" }, k === 0 ? "lowest" : k === NBIN - 1 ? "highest" : b); }));
    var body = [];
    cs.forEach(function (c) {
      var B5 = binsOf(F, c);
      if (B5.n < 50) return;
      var tr = h("tr", { class: c === OM.x ? "on" : null });
      tr.append(h("th", { scope: "row" }, CH[c][0]));
      for (var b = 0; b < NBIN; b++) {
        var means = [];
        field.forEach(function (k) {
          var sm = 0, cnt = 0, tot = 0;
          for (var i = 0; i < n; i++) if (B5.bin[i] === b) { tot++; var v = V[k][i]; if (isFinite(v)) { sm += v; cnt++; } }
          if (cnt >= 0.9 * tot && cnt) means.push([k, sm / cnt]);
        });
        means.sort(function (x, y) { return up ? y[1] - x[1] : x[1] - y[1]; });
        var pos = means.findIndex(function (x) { return x[0] === OM.r; }), td;
        if (pos < 0) { td = h("td", { class: "n na" }, "–"); }
        else {
          var sm2 = 0, c2 = 0, sv = 0;
          for (var i2 = 0; i2 < n; i2++) if (B5.bin[i2] === b && isFinite(med[i2]) && isFinite(V[OM.r][i2])) { sm2 += med[i2]; sv += V[OM.r][i2]; c2++; }
          var dlt = c2 ? (up ? (sv - sm2) / c2 : (sm2 - sv) / c2) : NaN, frac = means.length > 1 ? 1 - pos / (means.length - 1) : 1;
          var lo = B5.edges[b], hi = B5.edges[b + 1];
          td = h("td", { class: "n cellr", style: "--a:" + (0.06 + 0.8 * frac).toFixed(3), title: CH[c][0] + " from " + lo.toFixed(2) + " to " + hi.toFixed(2) + ": rank " + (pos + 1) + " of " + means.length + ", " + (dlt >= 0 ? "+" : "") + dlt.toFixed(OM.m === "psnr" ? 2 : 3) + " against the median method" },
            h("span", { class: "rk" + (frac > 0.5 ? " inv" : "") }, "#" + (pos + 1)), h("span", { class: "dl" + (frac > 0.5 ? " inv" : "") }, isFinite(dlt) ? (dlt >= 0 ? "+" : "−") + Math.abs(dlt).toFixed(OM.m === "psnr" ? 2 : 3) : ""));
        }
        tr.append(td);
      }
      tr.addEventListener("click", function () { OM.x = c; omX.value = c; drawOne(); });
      body.push(tr);
    });
    box.replaceChildren(h("div", { class: "scroll" }, h("table", { class: "prof" }, h("thead", null, head), h("tbody", null, body))));
    document.getElementById("om-p-note").textContent = "Frames of the set split into fifths along each characteristic. A cell is the rank of " + rowName(OM.r) + " by mean " + METRIC[OM.m].label +
      " among the GoPro-trained methods in that fifth (darker is higher), and its mean difference to the median method. Hover a cell for the range, click a row to plot it.";
  }

  /* ================================================================ wiring */
  function all() { drawPairs(); drawOne(); renderViewer(); }
  document.addEventListener("dbis:set", all);
  var lastW = document.documentElement.clientWidth, pend = 0;
  new ResizeObserver(function () {
    var w = document.documentElement.clientWidth;
    if (Math.abs(w - lastW) < 8) return;
    lastW = w; cancelAnimationFrame(pend);
    pend = requestAnimationFrame(function () { drawPairs(); drawOne(); drawStage(); });
  }).observe(document.body);
  drawPairs();
  drawOne();
})();
