import glob, json, re, sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

E = [json.loads(l) for l in open("data/eval/eval.jsonl", encoding="utf-8")]
O = [r for r in E if r["variant"] == "orig"]
H = [r for r in O if r["label"] == "H"]
DEV = set(json.load(open("kaggle/out_det_final_A/det_meta.json"))["dev_clusters"])
HELD = sorted({r["generator"] for r in O if r.get("heldout")})
KZ = [g for g in HELD if "KazLLM" in g or "Sherkala" in g]
GM = [g for g in HELD if g not in KZ]
INDIST = sorted({r["generator"] for r in O if r["label"] == "M" and r["generator"] not in HELD})
Hdev = [r for r in H if r["src_split"] == "train" and r["cluster"] in DEV]
Hte = [r for r in H if r["split"] == "test"]
Mid = [r for r in O if r["label"] == "M" and r["split"] == "test" and not r["heldout"]]
Mho = [r for r in O if r["label"] == "M" and r["heldout"] and r["src_split"] == "test"]
OUT = {}

from datasketch import MinHash, MinHashLSH

def mhash(t):
    w = re.findall(r"\w+", t.lower())
    m = MinHash(num_perm=128, seed=1)
    for i in range(max(1, len(w) - 4)):
        m.update(" ".join(w[i:i + 5]).encode("utf-8"))
    return m

MH = {r["id"]: mhash(r["text"]) for r in H}
lsh = MinHashLSH(threshold=0.3, num_perm=128)
for r in H:
    if r["split"] == "train":
        lsh.insert(r["id"], MH[r["id"]])
near = {r["id"] for r in H if r["split"] == "test" and any(MH[r["id"]].jaccard(MH[c]) >= 0.3 for c in lsh.query(MH[r["id"]]))}
cl = lambda sp, ess: {r["cluster"] for r in H if r["split"] == sp and (r["domain"] == "essay") == ess}
OUT["split"] = {"jaccard": 0.3, "test_humans_with_train_near_duplicate": len(near), "train_clusters": len(cl("train", False) | cl("train", True)),
                "test_clusters": len(cl("test", False) | cl("test", True)), "essay_train_clusters": len(cl("train", True)), "essay_test_clusters": len(cl("test", True)),
                "shared_clusters": len((cl("train", False) | cl("train", True)) & (cl("test", False) | cl("test", True))),
                "dev_clusters": len(DEV), "dev_humans_by_domain": {d: sum(r["domain"] == d for r in Hdev) for d in ("encyclopedic", "news", "web", "essay")}}

rng = np.random.default_rng(0)
n = len(Hdev)
true_fpr = np.array([1 - np.quantile(rng.random(n), 0.99) for _ in range(20000)])
OUT["threshold_theory"] = {"n": n, "mean": float(true_fpr.mean()), "lo": float(np.percentile(true_fpr, 2.5)), "hi": float(np.percentile(true_fpr, 97.5)),
                           "p_above_1.5pct": float(np.mean(true_fpr > 0.015))}

PUN = [",", ".", ";", ":", "!", "?", "\"", "'", "(", ")", "-", "%"]

def feats(t):
    w = t.split()
    nw = max(1, len(w))
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    sl = [len(s.split()) for s in sents] or [0]
    letters = [c for c in t if c.isalpha()]
    f = [len(w), len(t) / nw, t.count("\n\n"), len(t.split("\n")), float(np.mean(sl)), float(np.std(sl)),
         sum(c.isdigit() for c in t) / max(1, len(t)), sum(c.isupper() for c in letters) / max(1, len(letters)),
         sum(("a" <= c.lower() <= "z") for c in letters) / max(1, len(letters)), float(t.rstrip()[-1:] in ".!?\"") if t.strip() else 0.0,
         len(t.split("\n")[0].split())]
    return f + [t.count(p) / nw for p in PUN]

tr = [r for r in O if r["src_split"] == "train" and r["cluster"] not in DEV and (r["label"] == "H" or (r["label"] == "M" and r["generator"] in INDIST))]
X = np.array([feats(r["text"]) for r in tr])
sc = StandardScaler().fit(X)
lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), np.array([r["label"] == "M" for r in tr]))
P = lambda rows: lr.predict_proba(sc.transform(np.array([feats(r["text"]) for r in rows])))[:, 1]
pH, pM, pMh = P(Hte), P(Mid), P(Mho)
th = float(np.quantile(P(Hdev), 0.99))
mid_c = [i for i, r in enumerate(Mid) if 120 <= len(r["text"].split()) < 220]
hte_c = [i for i, r in enumerate(Hte) if 120 <= len(r["text"].split()) < 220]
auc = lambda p, nn: float(roc_auc_score([1] * len(p) + [0] * len(nn), list(p) + list(nn)))
OUT["surface"] = {"n_features": int(X.shape[1]), "auroc": auc(pM, pH), "auroc_heldout": auc(pMh, pH), "auroc_len120_220": auc(pM[mid_c], pH[hte_c]),
                  "fpr_at_dev": float(np.mean(pH >= th)), "tpr_at_dev": float(np.mean(pM >= th)), "tpr_heldout_at_dev": float(np.mean(pMh >= th))}

S = defaultdict(lambda: defaultdict(list))
for d in ("kaggle/out_det_final_A", "kaggle/out_det_final_B"):
    for fn in glob.glob(f"{d}/det_*_s*.jsonl"):
        tag = re.sub(r"_s\d+$", "", fn.split("/")[-1][:-6])
        for l in open(fn):
            x = json.loads(l)
            S[tag][x["id"]].append(x["p_machine"])
S = {t: {i: float(np.mean(v)) for i, v in m.items()} for t, m in S.items()}
S["bino"] = {json.loads(l)["id"]: -json.loads(l)["binoculars"] for l in open("results/zs_merged_F/zs_gemma4-e2b.jsonl")}
ra = np.sort([S["det_auglen"][r["id"]] for r in Hdev]); rb = np.sort([S["bino"][r["id"]] for r in Hdev])
S["combo"] = {i: max(np.searchsorted(ra, S["det_auglen"][i], "right") / len(ra), np.searchsorted(rb, S["bino"][i], "right") / len(rb)) for i in S["det_auglen"] if i in S["bino"]}
thr = {t: float(np.quantile([S[t][r["id"]] for r in Hdev], 0.99)) for t in ("det_base", "det_auglen", "combo")}

dig = lambda r: bool(re.search(r"\d", r["text"]))
H0 = [r for r in Hte if not dig(r)]; H1 = [r for r in Hte if dig(r)]; M0 = [r for r in Mid if not dig(r)]
OUT["digits"] = {"human_share": len(H1) / len(Hte), "machine_share": float(np.mean([dig(r) for r in Mid])), "n_human_no_digit": len(H0),
                 "n_human_no_digit_essay": sum(r["domain"] == "essay" for r in H0), "n_machine_no_digit": len(M0)}
for t in ("det_base", "combo"):
    OUT["digits"][t] = {"auroc_no_digit": auc([S[t][r["id"]] for r in M0], [S[t][r["id"]] for r in H0]),
                        "flagged_no_digit": int(sum(S[t][r["id"]] >= thr[t] for r in H0)), "fpr_no_digit": float(np.mean([S[t][r["id"]] >= thr[t] for r in H0])),
                        "fpr_digit": float(np.mean([S[t][r["id"]] >= thr[t] for r in H1]))}

fam = {}
for t in ("det_auglen", "combo"):
    hit = lambda gs: float(np.mean([S[t][r["id"]] >= thr[t] for r in Mho if r["generator"] in gs]))
    fam[t] = {"kazakh_built": hit(KZ), "gemma4": hit(GM)}
fam["n_kazakh_built"] = sum(r["generator"] in KZ for r in Mho)
fam["n_gemma4"] = sum(r["generator"] in GM for r in Mho)
OUT["family"] = fam

C = json.load(open("results/analysis_det.json"))["configs"]
NS = [k for k in C if k in ("det_base", "det_xlmr", "det_kznoaug") or k.startswith(("zs:", "ots_"))]
lat = {k: (C[k]["script"]["latin_tpr@dev"], C[k]["script"]["latin_fpr@dev"]) for k in NS}
low = {k: v for k, v in lat.items() if v[1] < 0.10}
high = {k: v for k, v in lat.items() if v[1] >= 0.10}
OUT["latin"] = {"n_detectors": len(lat), "max_tpr_when_fpr_below_10": max(v[0] for v in low.values()), "n_fpr_above_10": len(high),
                "min_fpr_of_those": min(v[1] for v in high.values()), "max_tpr_of_those": max(v[0] for v in high.values()),
                "by_detector": {k: {"tpr": v[0], "fpr": v[1]} for k, v in lat.items()}}

sys.path.insert(0, "scripts")
import prompts as PR
POOLS = {"C1_ENC": PR.C1_ENC, "C1_GEN": PR.C1_GEN, "C1_ESSAY": PR.C1_ESSAY, "C2": PR.C2, "C3": PR.C3, "C4": PR.C4, "C6": PR.C6}
TRX = [(sp, re.compile(re.sub(r"\\\{\w+\\\}", lambda _: "(?s:.*?)", re.escape(t)), re.S)) for v in POOLS.values() for sp in ("train", "test") for t in v[sp]]
gn = lambda g: re.sub(r"[^a-z0-9]", "", g.split(":")[-1].split("/")[-1].lower())
used = defaultdict(set)
for r in O:
    if r["label"] != "H":
        used[(r["condition"], r["group"])].add(gn(r["generator"]))
tp = Counter()
seen = set()
for fn in sorted(glob.glob("data/machine/*.tasks.jsonl") + glob.glob("kaggle/gen_inputs/*.tasks.jsonl")):
    for l in open(fn, encoding="utf-8"):
        t = json.loads(l)
        if "prompt" not in t or t["task_id"] in seen or not any(g in gn(t["generator"]) or gn(t["generator"]) in g for g in used.get((t["condition"], t["group"]), ())):
            continue
        seen.add(t["task_id"])
        sps = [sp for sp, rx in TRX if rx.fullmatch(t["prompt"])]
        tp["own pool only" if sps == [t["split"]] else "other or several"] += 1
OUT["templates"] = {"prompts_own_pool_only": tp["own pool only"], "prompts_other": tp["other or several"],
                    "identical_templates_across_pools": sum(len(set(v["train"]) & set(v["test"])) for v in POOLS.values())}

pool = lambda gens: [r for r in O if r["src_split"] == "train" and r["cluster"] not in DEV and (r["label"] == "H" or (r["label"] == "M" and r["generator"] in gens))]
P1, P2 = pool(INDIST), pool(INDIST + KZ)
OUT["pools"] = {"main_n": len(P1), "main_heldout_rows": sum(r["generator"] in HELD for r in P1), "main_test_rows": sum(r["split"] == "test" and r["label"] == "H" for r in P1),
                "kz_n": len(P2), "kz_rows": sum(r["generator"] in KZ for r in P2), "kz_gemma_rows": sum(r["generator"] in GM for r in P2),
                "dev_rows_any": sum(r["cluster"] in DEV for r in P1 + P2)}

ids = {r["id"] for r in H}
dates = defaultdict(list)
for fn in ("data/human/human_clean.jsonl", "data/human/essay_clean.jsonl"):
    for l in open(fn, encoding="utf-8"):
        h = json.loads(l)
        if h["id"] in ids:
            dates[h["domain"]].append(h.get("rev_timestamp") or h.get("crawl_date"))
OUT["dates"] = {d: {"n": len(v), "dated": sum(x is not None for x in v), "max": max(x for x in v if x) if any(v) else None} for d, v in dates.items()}
json.dump(OUT, open("results/validity.json", "w"), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k != "latin"}, indent=1))
print({k: v for k, v in OUT["latin"].items() if k != "by_detector"})
