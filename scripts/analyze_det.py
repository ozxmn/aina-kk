import glob, json, math, os, re, sys, random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
DIRS, OUTF = sys.argv[1:-1], sys.argv[-1]
E = {r["id"]: r for r in map(json.loads, open("data/eval/eval.jsonl", encoding="utf-8"))}
for r in E.values(): r["n_words"] = r.get("n_words") or len(r.get("text", "").split())
metas = [f"{d}/det_meta.json" for d in DIRS if os.path.exists(f"{d}/det_meta.json")]
meta = json.load(open(metas[0])) if metas else {"dev_clusters": []}
assert all(json.load(open(m))["dev_clusters"] == meta["dev_clusters"] for m in metas), "dev clusters differ between runs"
DEV_CL = set(meta["dev_clusters"])
S = defaultdict(lambda: defaultdict(list))
SEED = defaultdict(dict)
for RD in DIRS:
    for fn in glob.glob(f"{RD}/det_*_s*.jsonl") + glob.glob(f"{RD}/ots_*.jsonl"):
        b = os.path.basename(fn)[:-6]
        tag = re.sub(r"_s\d+$", "", b)
        for l in open(fn):
            d = json.loads(l)
            if math.isfinite(d["p_machine"]): S[tag][d["id"]].append(d["p_machine"]); SEED[tag].setdefault(b, {})[d["id"]] = d["p_machine"]
    for fn in glob.glob(f"{RD}/zs_*.jsonl"):
        sc = os.path.basename(fn)[3:-6]
        for l in open(fn):
            d = json.loads(l)
            for k, sign in (("binoculars", -1), ("obs_fdg", 1), ("perf_fdg", 1)):
                if math.isfinite(d[k]): S[f"zs:{sc}:{k}"][d["id"]].append(sign * d[k])
S = {t: {i: float(np.mean(v)) for i, v in dd.items()} for t, dd in S.items()}
def sel(f): return [r for r in E.values() if f(r)]
H_dev = sel(lambda r: r["label"] == "H" and r["variant"] == "orig" and r["src_split"] == "train" and r["cluster"] in DEV_CL)
H_tr = sel(lambda r: r["label"] == "H" and r["variant"] == "orig" and r["src_split"] == "train")
H_te = sel(lambda r: r["label"] == "H" and r["variant"] == "orig" and r["split"] == "test")
M_id = sel(lambda r: r["label"] == "M" and r["variant"] == "orig" and r["split"] == "test" and not r["heldout"])
M_ho = sel(lambda r: r["label"] == "M" and r["variant"] == "orig" and r["heldout"] and r["src_split"] == "test")
GENS = sorted({r["generator"] for r in E.values() if r["label"] == "M"})
GRAY = {g: sel(lambda r, g=g: r["label"] == g and r["variant"] == "orig" and (g == "MT" or r["src_split"] == "test")) for g in ("HP", "MT", "MIX", "MH")}
def _ecdf(t, ref):
    rv = np.sort(np.array([S[t][r["id"]] for r in ref if r["id"] in S[t]]))
    return {i: float(np.searchsorted(rv, x, side="right")) / len(rv) for i, x in S[t].items()}
PARTS = {}
for a in ("det_base", "det_aug", "det_auglen", "det_all", "det_alllen"):
    b = "zs:gemma4-e2b:binoculars"
    if a in S and b in S and H_dev:
        ma, mb = _ecdf(a, H_dev), _ecdf(b, H_dev)
        S[f"combo_{a[4:]}+bino"] = {i: max(ma[i], mb[i]) for i in ma if i in mb}
        PARTS[f"combo_{a[4:]}+bino"] = (a, b)
BINS = {"120-160": (0, 160), "160-220": (160, 220), "220-320": (220, 10**6)}
def vals(t, rows): s = S[t]; return [s[r["id"]] for r in rows if r["id"] in s]
def auc(t, pos, neg):
    p, n = vals(t, pos), vals(t, neg)
    return None if len(p) < 5 or len(n) < 5 else float(roc_auc_score([1] * len(p) + [0] * len(n), p + n))
def thr(t, neg, fpr=0.01):
    v = vals(t, neg); return float(np.quantile(v, 1 - fpr)) if v else None
def rate(t, rows, th):
    v = vals(t, rows); return float(np.mean([x >= th for x in v])) if v and th is not None else None
def rate_lenaware(t, rows, ths):
    s = S[t]; hits = []
    for r in rows:
        if r["id"] not in s: continue
        b = next(k for k, (lo, hi) in BINS.items() if lo <= r["n_words"] < hi)
        if ths[b] is None: continue
        hits.append(s[r["id"]] >= ths[b])
    return float(np.mean(hits)) if hits else None
def boot_rate(t, rows, th, B=400, seed=0):
    rng = random.Random(seed); g = defaultdict(list)
    for r in rows: g[r["group"]].append(r)
    keys = list(g); out = []
    for _ in range(B):
        rs = [x for k in (rng.choice(keys) for _ in keys) for x in g[k]]
        v = rate(t, rs, th)
        if v is not None: out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else None
def boot_thr(t, rows, cal_map=None, B=400, seed=0):
    rng = random.Random(seed); dev = [r["id"] for r in H_dev]
    g = defaultdict(list)
    for r in rows: g[r["group"]].append(r["id"])
    keys = list(g); out = []
    for _ in range(B):
        ds = [rng.choice(dev) for _ in dev]
        ts = [x for k in (rng.choice(keys) for _ in keys) for x in g[k]]
        if t in PARTS:
            a, b = PARTS[t]
            ra, rb = np.sort([S[a][i] for i in ds]), np.sort([S[b][i] for i in ds])
            f = lambda ids: np.maximum(np.searchsorted(ra, [S[a][i] for i in ids], "right") / len(ra), np.searchsorted(rb, [S[b][i] for i in ids], "right") / len(rb))
        else:
            f = lambda ids: np.array([S[t][i] for i in ids])
        th = float(np.quantile(f([cal_map[i] for i in ds] if cal_map else ds), 0.99))
        out.append(float(np.mean(f(ts) >= th)))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
def boot_auc(t, pos, neg, B=300, seed=0):
    rng = random.Random(seed); g = defaultdict(list)
    for r in pos: g[r["group"]].append((r, 1))
    for r in neg: g[r["group"]].append((r, 0))
    keys = list(g); out = []; s = S[t]
    for _ in range(B):
        rs = [x for k in (rng.choice(keys) for _ in keys) for x in g[k] if x[0]["id"] in s]
        ys = [y for _, y in rs]
        if 0 < sum(ys) < len(ys): out.append(roc_auc_score(ys, [s[r["id"]] for r, _ in rs]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else None
res = {"n": {"dev_clusters": len(DEV_CL), "H_dev": len(H_dev), "H_train_zs": len(H_tr), "H_test": len(H_te), "M_indist": len(M_id), "M_heldout_topicdisjoint": len(M_ho)}, "configs": {}}
for t in sorted(S):
    if not S[t]: continue
    e = {}
    e["auroc"] = auc(t, M_id, H_te); e["auroc_heldout"] = auc(t, M_ho, H_te)
    e["auroc_ci"] = boot_auc(t, M_id, H_te) if e["auroc"] is not None else None
    Hcal = H_tr if t.startswith("zs:") else H_dev
    th = thr(t, Hcal) if Hcal else None
    e["thr_dev"] = th
    d05 = None if t.startswith(("zs:", "combo_")) else 0.5
    for name, x in (("0.5", d05), ("dev", th)):
        e[f"fpr@{name}"] = rate(t, H_te, x); e[f"tpr@{name}"] = rate(t, M_id, x); e[f"tpr_heldout@{name}"] = rate(t, M_ho, x)
        e[f"fpr_by_domain@{name}"] = {d: rate(t, [r for r in H_te if r["domain"] == d], x) for d in sorted({r["domain"] for r in H_te})}
        e[f"fpr_by_len@{name}"] = {b: rate(t, [r for r in H_te if lo <= r["n_words"] < hi], x) for b, (lo, hi) in BINS.items()}
        e[f"flag_gray@{name}"] = {g: rate(t, GRAY[g], x) for g in ("HP", "MT", "MIX", "MH")}
        e[f"flag_by_gen@{name}"] = {g: rate(t, [r for r in M_id + M_ho if r["generator"] == g], x) for g in GENS}
    e["fpr@dev_ci"] = boot_rate(t, H_te, th) if th is not None else None
    if t.startswith(("det_", "combo_")) and not t.startswith("det_logo"):
        e["fpr@dev_ci_thr"] = boot_thr(t, H_te)
    if Hcal:
        ths = {b: thr(t, [r for r in Hcal if lo <= r["n_words"] < hi]) for b, (lo, hi) in BINS.items()}
        e["thr_lenaware"] = ths
        e["fpr_by_len@lenaware"] = {b: rate(t, [r for r in H_te if lo <= r["n_words"] < hi], ths[b]) for b, (lo, hi) in BINS.items()}
        e["fpr@lenaware"] = rate_lenaware(t, H_te, ths); e["tpr@lenaware"] = rate_lenaware(t, M_id, ths)
    e["auroc_by_gen"] = {g: auc(t, [r for r in M_id + M_ho if r["generator"] == g], H_te) for g in GENS}
    v = lambda lab, var: sel(lambda r: r["label"] == lab and r["variant"] == var)
    e["script"] = {"latin2021": auc(t, v("M", "latin2021"), v("H", "latin2021")), "latin2021canon": auc(t, v("M", "latin2021canon"), v("H", "latin2021canon")),
                   "trunc100": auc(t, v("M", "trunc100"), v("H", "trunc100")), "trunc50": auc(t, v("M", "trunc50"), v("H", "trunc50"))}
    base_hg = [E[r["base_id"]] for r in v("M", "homoglyph10")]
    e["script"]["hg_tpr_orig@dev"] = rate(t, base_hg, th); e["script"]["hg_tpr@dev"] = rate(t, v("M", "homoglyph10"), th)
    e["script"]["hgfold_tpr@dev"] = rate(t, v("M", "homoglyph10fold"), th)
    e["script"]["latin_fpr@dev"] = rate(t, v("H", "latin2021"), th); e["script"]["latin_tpr@dev"] = rate(t, v("M", "latin2021"), th)
    e["script"]["canon_fpr@dev"] = rate(t, v("H", "latin2021canon"), th); e["script"]["canon_tpr@dev"] = rate(t, v("M", "latin2021canon"), th)
    fo = [r for r in v("H", "folded") if r["split"] == "test"]
    e["script"]["hghum_n"] = len(fo)
    e["script"]["hghum_raw@dev"] = rate(t, [E[r["base_id"]] for r in fo], th); e["script"]["hghum_fold@dev"] = rate(t, fo, th)
    sh = {}
    for n in ("100", "50"):
        Ht, Mt = v("H", "trunc" + n), v("M", "trunc" + n)
        sh[n] = {"auroc": auc(t, Mt, Ht), "fpr@0.5": rate(t, Ht, d05), "tpr@0.5": rate(t, Mt, d05),
                 "fpr@dev": rate(t, Ht, th), "tpr@dev": rate(t, Mt, th)}
        Hd = v("H", "devtrunc" + n)
        if len(vals(t, Hd)) >= 50:
            thn = thr(t, Hd); sh[n]["thr_devlen"] = thn
            sh[n]["fpr@devlen"] = rate(t, Ht, thn); sh[n]["tpr@devlen"] = rate(t, Mt, thn)
            sh[n]["fpr@devlen_ci"] = boot_rate(t, Ht, thn)
            if t.startswith(("det_", "combo_")) and not t.startswith("det_logo"):
                cm = {r["base_id"]: r["id"] for r in Hd}
                sh[n]["fpr@devlen_ci_thr"] = boot_thr(t, Ht, cal_map=cm)
    e["short"] = sh
    e["auroc_len120_220"] = auc(t, [r for r in M_id if 120 <= r["n_words"] < 220], [r for r in H_te if 120 <= r["n_words"] < 220])
    if t.startswith("det_") and len(SEED.get(t, {})) > 1:
        sp = defaultdict(list)
        for b, sd in SEED[t].items():
            hv = [sd[r["id"]] for r in H_dev if r["id"] in sd]; th1 = float(np.quantile(hv, 0.99))
            f1 = lambda rows: float(np.mean([sd[r["id"]] >= th1 for r in rows if r["id"] in sd]))
            sp["fpr@dev"].append(f1(H_te)); sp["tpr@dev"].append(f1(M_id)); sp["tpr_heldout@dev"].append(f1(M_ho))
            sp["fpr50@dev"].append(f1(v("H", "trunc50")))
            for g in GENS: sp["gen/" + g].append(f1([r for r in M_id + M_ho if r["generator"] == g]))
        e["seed_range"] = {k: [min(x), max(x)] for k, x in sp.items() if not k.startswith("gen/")}
        e["seed_range"]["gen"] = {k[4:]: [min(x), max(x)] for k, x in sp.items() if k.startswith("gen/")}
    res["configs"][t] = e
logo = {}
COMP = "det_auglen" if "det_auglen" in S else "det_all"
for g in GENS:
    t = "det_logo-" + g
    if t not in S: continue
    Mg = [r for r in M_id + M_ho if r["generator"] == g]
    th = thr(t, H_dev)
    logo[g] = {"auroc": auc(t, Mg, H_te), "tpr@dev": rate(t, Mg, th), "fpr@dev": rate(t, H_te, th), "comparator": COMP,
               "auroc_when_trained": auc(COMP, Mg, H_te) if COMP in S else None,
               "tpr@dev_when_trained": rate(COMP, Mg, thr(COMP, H_dev)) if COMP in S else None}
res["logo"] = logo
comp = {}
zb = "zs:gemma4-e2b:binoculars"
KZ = [r for r in M_ho if "KazLLM" in r["generator"] or "Sherkala" in r["generator"]]
for a in ("det_base", "det_aug", "det_auglen"):
    if a in S and zb in S:
        ta, tb = thr(a, H_dev), thr(zb, H_tr)
        missed = [r for r in KZ if r["id"] in S[a] and r["id"] in S[zb] and S[a][r["id"]] < ta]
        comp[a[4:]] = {"n_missed": len(missed), "bino_catches": float(np.mean([S[zb][r["id"]] >= tb for r in missed])) if missed else None,
                       "n_kz": len(KZ)}
res["complement"] = comp
json.dump(res, open(OUTF, "w"), indent=1)
for t, e in res["configs"].items():
    print(f"{t:28s} auc {e['auroc'] and round(e['auroc'],3)} held {e['auroc_heldout'] and round(e['auroc_heldout'],3)} "
          f"fpr@.5 {e['fpr@0.5'] and round(e['fpr@0.5'],3)} fpr@dev {e['fpr@dev'] and round(e['fpr@dev'],3)} tpr@dev {e['tpr@dev'] and round(e['tpr@dev'],3)} "
          f"latin {e['script']['latin2021'] and round(e['script']['latin2021'],3)} canon {e['script']['latin2021canon'] and round(e['script']['latin2021canon'],3)}")
for g, x in logo.items(): print("LOGO", g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
