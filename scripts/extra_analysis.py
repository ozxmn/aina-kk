import glob, json, math, os, random, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import roc_auc_score

DIRS, OUTF = sys.argv[1:-1], sys.argv[-1]
E = {r["id"]: r for r in map(json.loads, open("data/eval/eval.jsonl", encoding="utf-8"))}
meta = json.load(open(next(f"{d}/det_meta.json" for d in DIRS if os.path.exists(f"{d}/det_meta.json"))))
DEV_CL = set(meta["dev_clusters"])

S = defaultdict(lambda: defaultdict(list))
ZS_KEYS = ["binoculars"] + [f"{w}_{k}" for w in ("obs", "perf") for k in ("ll", "logrank", "entropy", "lrr", "fdg")]
for RD in DIRS:
    for fn in glob.glob(f"{RD}/det_*_s*.jsonl"):
        tag = re.sub(r"_s\d+$", "", os.path.basename(fn)[:-6])
        for l in open(fn):
            d = json.loads(l)
            if math.isfinite(d["p_machine"]): S[tag][d["id"]].append(d["p_machine"])
    for fn in glob.glob(f"{RD}/zs_*.jsonl"):
        sc = os.path.basename(fn)[3:-6]
        for l in open(fn):
            d = json.loads(l)
            for k in ZS_KEYS:
                if k in d and d[k] is not None and math.isfinite(d[k]): S[f"zsraw:{sc}:{k}"][d["id"]].append(d[k])
S = {t: {i: float(np.mean(v)) for i, v in dd.items()} for t, dd in S.items()}
for k in list(S):
    if k.startswith("zsraw:") and k.endswith(":binoculars"):
        S["zs:" + k.split(":")[1] + ":binoculars"] = {i: -x for i, x in S[k].items()}

def sel(f): return [r for r in E.values() if f(r)]
H_dev = sel(lambda r: r["label"] == "H" and r["variant"] == "orig" and r["src_split"] == "train" and r["cluster"] in DEV_CL)
H_tr = sel(lambda r: r["label"] == "H" and r["variant"] == "orig" and r["src_split"] == "train")
H_te = sel(lambda r: r["label"] == "H" and r["variant"] == "orig" and r["split"] == "test")
M_id = sel(lambda r: r["label"] == "M" and r["variant"] == "orig" and r["split"] == "test" and not r["heldout"])
M_ho = sel(lambda r: r["label"] == "M" and r["variant"] == "orig" and r["heldout"] and r["src_split"] == "test")
M_tr = sel(lambda r: r["label"] == "M" and r["variant"] == "orig" and r["src_split"] == "train" and not r["heldout"] and r["cluster"] not in DEV_CL)
isKZ = lambda g: "KazLLM" in g or "Sherkala" in g
KZ = [r for r in M_ho if isKZ(r["generator"])]
GM = [r for r in M_ho if not isKZ(r["generator"])]
KAZ = [r for r in KZ if "KazLLM" in r["generator"]]
SHE = [r for r in KZ if "Sherkala" in r["generator"]]
Hd = {n: sel(lambda r, n=n: r["label"] == "H" and r["variant"] == "devtrunc" + n) for n in ("100", "50")}
BINO = "zs:gemma4-e2b:binoculars"

def ecdf_map(t, ref_ids):
    rv = np.sort([S[t][i] for i in ref_ids if i in S[t]])
    return lambda x: float(np.searchsorted(rv, x, side="right")) / len(rv)

def combo_scorer(a, ref_ids):
    fa, fb = ecdf_map(a, ref_ids), ecdf_map(BINO, ref_ids)
    return lambda i: max(fa(S[a][i]), fb(S[BINO][i]))

def scorer(t, ref_ids):
    if t.startswith("combo:"):
        return combo_scorer(t.split(":", 1)[1], ref_ids)
    return lambda i: S[t][i]

def q99(v): return float(np.quantile(v, 0.99))
def has(t, i):
    if t.startswith("combo:"):
        a = t.split(":", 1)[1]; return i in S[a] and i in S[BINO]
    return i in S[t]

def thresholds(t, dev_full, dev_trunc):
    ref = [r["id"] for r in dev_full]
    f = scorer(t, ref)
    th = {"full": q99([f(i) for i in ref if has(t, i)])}
    for n, rows in dev_trunc.items():
        th[n] = q99([f(r["id"]) for r in rows if has(t, r["id"])])
    return f, th

def bucket(n_words):
    return "full" if n_words >= 110 else "100" if n_words >= 100 else "50" if n_words >= 50 else None

def rate_rule(f, th, rows, t, rule):
    hits = []
    for r in rows:
        if not has(t, r["id"]): continue
        b = "full" if rule == "paper" else bucket(r["n_words"])
        if b is None: continue
        hits.append(f(r["id"]) >= th[b])
    return hits

def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]

def boot(t, rows, rule, B=1000, seed=0):
    rng = random.Random(seed)
    dev_ids = [r["id"] for r in H_dev]
    trunc_by_base = {n: {r["base_id"]: r for r in Hd[n]} for n in Hd}
    groups = defaultdict(list)
    for r in rows: groups[r["group"]].append(r)
    keys = list(groups); out = []
    for _ in range(B):
        ds = [rng.choice(dev_ids) for _ in dev_ids]
        dfull = [E[i] for i in ds]
        dtr = {n: [trunc_by_base[n][i] for i in ds if i in trunc_by_base[n]] for n in Hd}
        f, th = thresholds(t, dfull, dtr if rule == "deployed" else {})
        rs = [x for k in (rng.choice(keys) for _ in keys) for x in groups[k]]
        h = rate_rule(f, th, rs, t, rule)
        if h: out.append(float(np.mean(h)))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

OUT = {"n": {"H_dev": len(H_dev), "H_test": len(H_te), "M_indist": len(M_id), "M_heldout": len(M_ho), "KZ": len(KZ),
             "KazLLM": len(KAZ), "Sherkala": len(SHE), "Gemma4": len(GM)}}

SUBSETS = {"fpr": H_te, "tpr_indist": M_id, "tpr_heldout": M_ho, "kz": KZ, "kazllm": KAZ, "sherkala": SHE, "gemma4": GM}
dep = {}
for t in ("det_auglen", "combo:det_auglen", "combo:det_aug", "det_base"):
    f, th = thresholds(t, H_dev, Hd)
    e = {"thresholds": th}
    for rule in ("paper", "deployed"):
        e[rule] = {}
        for name, rows in SUBSETS.items():
            h = rate_rule(f, th, rows, t, rule)
            e[rule][name] = {"k": int(sum(h)), "n": len(h), "rate": float(np.mean(h))}
    e["routed_short"] = {name: float(np.mean([bucket(r["n_words"]) != "full" for r in rows])) for name, rows in SUBSETS.items()}
    e["fpr_ci_thr"] = {rule: boot(t, H_te, rule, B=400) for rule in ("deployed",)}
    dep[t] = e
OUT["deployed"] = dep

def auc(pos, neg):
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg)))
zs = {}
for t in sorted(k for k in S if k.startswith("zsraw:")):
    _, fam, stat = t.split(":")
    s = S[t]
    tr_pos = [s[r["id"]] for r in M_tr if r["id"] in s]; tr_neg = [s[r["id"]] for r in H_tr if r["id"] in s]
    sign = 1.0 if auc(tr_pos, tr_neg) >= 0.5 else -1.0
    v = {i: sign * x for i, x in s.items()}
    th = q99([v[r["id"]] for r in H_tr if r["id"] in v])
    g = lambda rows: [v[r["id"]] for r in rows if r["id"] in v]
    zs[f"{fam}:{stat}"] = {"sign": sign, "auroc": auc(g(M_id), g(H_te)), "auroc_heldout": auc(g(M_ho), g(H_te)),
                           "fpr": float(np.mean([x >= th for x in g(H_te)])), "tpr": float(np.mean([x >= th for x in g(M_id)])),
                           "tpr_heldout": float(np.mean([x >= th for x in g(M_ho)]))}
OUT["zs_all"] = zs
OUT["zs_all_best"] = max(zs.items(), key=lambda kv: kv[1]["tpr"])[0]

ci = {}
for t in ("det_base", BINO, "det_auglen", "combo:det_auglen", "combo:det_aug", "det_kznoaug"):
    if t == BINO:
        f = scorer(t, [r["id"] for r in H_tr]); th = {"full": q99([f(r["id"]) for r in H_tr])}
    else:
        f, th = thresholds(t, H_dev, Hd)
    e = {}
    for name, rows in (("kazllm", KAZ), ("sherkala", SHE), ("kz", KZ), ("gemma4", GM)):
        h = rate_rule(f, th, rows, t, "paper"); k, n = int(sum(h)), len(h)
        e[name] = {"k": k, "n": n, "rate": k / n, "cp": cp(k, n)}
        if t != BINO:
            e[name]["boot"] = boot(t, rows, "paper", B=500)
    ci[t] = e
OUT["ci"] = ci

def flags(t, rows):
    f, th = thresholds(t, H_dev, Hd)
    return {r["id"]: f(r["id"]) >= th["full"] for r in rows if has(t, r["id"])}
mc = {}
for a, b in (("det_base", "combo:det_base"), ("det_aug", "combo:det_aug"), ("det_auglen", "combo:det_auglen"),
             ("combo:det_aug", "combo:det_auglen")):
    fa, fb = flags(a, KZ), flags(b, KZ)
    ids = [i for i in fa if i in fb]
    only_b = sum(fb[i] and not fa[i] for i in ids); only_a = sum(fa[i] and not fb[i] for i in ids)
    p = binomtest(only_b, only_b + only_a, 0.5).pvalue if only_a + only_b else 1.0
    mc[f"{a} vs {b}"] = {"n": len(ids), "only_first": int(only_a), "only_second": int(only_b), "p_exact": float(p)}
OUT["mcnemar_kz"] = mc

Horig = [r for r in E.values() if r["label"] == "H" and r["variant"] == "orig"]
OUT["misc"] = {"homoglyph_share_all_humans": float(np.mean([r.get("homoglyphs", 0) > 0 for r in Horig])),
               "homoglyph_share_test_humans": float(np.mean([r.get("homoglyphs", 0) > 0 for r in H_te])),
               "n_homoglyph_test_humans": int(sum(r.get("homoglyphs", 0) > 0 for r in H_te)),
               "short_full_passages": {"human_test_under110": int(sum(r["n_words"] < 110 for r in H_te)),
                                       "kz_under110": int(sum(r["n_words"] < 110 for r in KZ)),
                                       "kz_under100": int(sum(r["n_words"] < 100 for r in KZ))}}
json.dump(OUT, open(OUTF, "w"), indent=1)
print(json.dumps({k: OUT[k] for k in ("n", "misc", "zs_all_best", "mcnemar_kz")}, indent=1))
for t, e in dep.items():
    print(t, "thr", {k: round(v, 4) for k, v in e["thresholds"].items()})
    for rule in ("paper", "deployed"):
        print("  ", rule, {k: f"{v['k']}/{v['n']}={100*v['rate']:.1f}" for k, v in e[rule].items()})
    print("   fpr CI (deployed, threshold re-fit):", [round(100 * x, 1) for x in e["fpr_ci_thr"]["deployed"]], "routed short:", {k: round(v, 3) for k, v in e["routed_short"].items()})
for t, e in ci.items():
    print(t, {k: (f"{v['k']}/{v['n']}", [round(100*x,1) for x in v['cp']], [round(100*x,1) for x in v.get('boot',[])]) for k, v in e.items()})

def v(lab, var): return sel(lambda r: r["label"] == lab and r["variant"] == var)
GRAY = {g: sel(lambda r, g=g: r["label"] == g and r["variant"] == "orig" and (g == "MT" or r["src_split"] == "test")) for g in ("HP", "MT", "MIX", "MH")}
def ppv(tpr_id, tpr_ho, fpr, prev=0.05):
    t = (tpr_id * len(M_id) + tpr_ho * len(M_ho)) / (len(M_id) + len(M_ho))
    return prev * t / (prev * t + (1 - prev) * fpr)
rows4 = {}
for t in ("det_auglen", "combo:det_auglen", "det_kzlen", "combo:det_aug"):
    f, th = thresholds(t, H_dev, Hd)
    R = lambda rows: rate_rule(f, th, rows, t, "deployed")
    m = lambda rows: float(np.mean(R(rows)))
    e = {"fpr": m(H_te), "fpr_web": m([r for r in H_te if r["domain"] == "web"]), "fpr_essay": m([r for r in H_te if r["domain"] == "essay"]),
         "tpr": m(M_id), "tpr_heldout": m(M_ho), "kazllm": m(KAZ), "sherkala": m(SHE), "kz": m(KZ), "gemma4": m(GM),
         "latin_canon": m(v("M", "latin2021canon")), "latin_canon_fpr": m(v("H", "latin2021canon")),
         "hgfold": m(v("M", "homoglyph10fold")),
         "fpr50": float(np.mean([f(r["id"]) >= th["50"] for r in v("H", "trunc50") if has(t, r["id"])])),
         "tpr50": float(np.mean([f(r["id"]) >= th["50"] for r in v("M", "trunc50") if has(t, r["id"])])),
         "gray": {g: m(rows) for g, rows in GRAY.items()},
         "per_gen": {g: m([r for r in M_id + M_ho if r["generator"] == g]) for g in sorted({r["generator"] for r in M_id + M_ho})}}
    e["ppv5"] = ppv(e["tpr"], e["tpr_heldout"], e["fpr"])
    H0 = [r for r in H_te if not re.search(r"\d", r.get("text") or "")] if all("text" in r for r in H_te) else None
    rows4[t] = e
OUT["table4_deployed"] = rows4
mcd = {}
def flags_dep(t, rows):
    f, th = thresholds(t, H_dev, Hd)
    return {r["id"]: f(r["id"]) >= th[bucket(r["n_words"])] for r in rows if has(t, r["id"])}
for a, b in (("det_base", "combo:det_base"), ("det_aug", "combo:det_aug"), ("det_auglen", "combo:det_auglen"), ("combo:det_aug", "combo:det_auglen")):
    fa, fb = flags_dep(a, KZ), flags_dep(b, KZ)
    ids = [i for i in fa if i in fb]
    ob = sum(fb[i] and not fa[i] for i in ids); oa = sum(fa[i] and not fb[i] for i in ids)
    mcd[f"{a} vs {b}"] = {"n": len(ids), "only_first": int(oa), "only_second": int(ob), "p_exact": float(binomtest(ob, ob + oa, 0.5).pvalue) if oa + ob else 1.0}
OUT["mcnemar_kz_deployed"] = mcd
json.dump(OUT, open(OUTF, "w"), indent=1)
for t, e in rows4.items():
    print("T4dep", t, {k: (round(100 * x, 1) if isinstance(x, float) else x) for k, x in e.items() if k not in ("gray", "per_gen")}, {g: round(100 * x, 1) for g, x in e["gray"].items()})
print(json.dumps(mcd, indent=1))

cid = {}
for t in ("det_auglen", "combo:det_auglen", "det_kzlen"):
    f, th = thresholds(t, H_dev, Hd)
    e = {"fpr_ci_thr": boot(t, H_te, "deployed", B=400)}
    for name, rows in (("kazllm", KAZ), ("sherkala", SHE), ("kz", KZ), ("gemma4", GM)):
        h = rate_rule(f, th, rows, t, "deployed"); k, n = int(sum(h)), len(h)
        e[name] = {"k": k, "n": n, "rate": k / n, "cp": cp(k, n), "boot": boot(t, rows, "deployed", B=500)}
    cid[t] = e
OUT["ci_deployed"] = cid
json.dump(OUT, open(OUTF, "w"), indent=1)
for t, e in cid.items():
    print("CIdep", t, [round(100*x,1) for x in e["fpr_ci_thr"]], {k: (f"{v['k']}/{v['n']}", [round(100*x,1) for x in v['cp']], [round(100*x,1) for x in v['boot']]) for k, v in e.items() if k != "fpr_ci_thr"})
