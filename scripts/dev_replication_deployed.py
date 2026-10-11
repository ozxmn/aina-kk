import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

E = {}
for l in open("data/eval/eval.jsonl", encoding="utf-8"):
    r = json.loads(l)
    E[r["id"]] = r
DIRS = sys.argv[1:] or ["kaggle/out_det_final_A", "kaggle/out_det_final_B", "kaggle/out_det_final_K", "results/zs_merged_F"]
DEV = set(json.load(open(next(f"{d}/det_meta.json" for d in DIRS if os.path.exists(f"{d}/det_meta.json"))))["dev_clusters"])
HELD = {r["generator"] for r in E.values() if r.get("heldout")}
KZ = {g for g in HELD if "KazLLM" in g or "Sherkala" in g}
GM = HELD - KZ

S = defaultdict(lambda: defaultdict(list))
for d in DIRS:
    for fn in glob.glob(f"{d}/det_*_s*.jsonl"):
        tag = re.sub(r"_s\d+$", "", fn.split("/")[-1][:-6])
        for l in open(fn):
            x = json.loads(l)
            S[tag][x["id"]].append(x["p_machine"])
S = {t: {i: float(np.mean(v)) for i, v in m.items()} for t, m in S.items()}
S["bino"] = {}
for l in open(next(f"{d}/zs_gemma4-e2b.jsonl" for d in DIRS if os.path.exists(f"{d}/zs_gemma4-e2b.jsonl"))):
    x = json.loads(l)
    S["bino"][x["id"]] = -x["binoculars"]

isdev = lambda r: r["src_split"] == "train" and r["cluster"] in DEV
Hd = [r for r in E.values() if r["label"] == "H" and r["variant"] == "orig" and isdev(r)]
H50 = [r for r in E.values() if r["label"] == "H" and r["variant"] == "devtrunc50" and isdev(r)]
H100 = [r for r in E.values() if r["label"] == "H" and r["variant"] == "devtrunc100" and isdev(r)]
BK = lambda n: "full" if n >= 110 else "100" if n >= 100 else "50"
Md = [r for r in E.values() if r["label"] == "M" and r["variant"] == "orig" and isdev(r)]
CL = sorted(DEV)

def ecdf(ref):
    rv = np.sort(np.array(ref))
    return lambda x: float(np.searchsorted(rv, x, side="right")) / len(rv)

def scorer(cfg, ref_rows):
    if not cfg.startswith("combo:"):
        return lambda i: S[cfg][i]
    a = cfg.split(":")[1]
    fa, fb = ecdf([S[a][r["id"]] for r in ref_rows]), ecdf([S["bino"][r["id"]] for r in ref_rows])
    return lambda i: max(fa(S[a][i]), fb(S["bino"][i]))

def q99(v):
    return float(np.quantile(v, 0.99))

def run(cfg, mode):
    hit = defaultdict(list)
    for c in CL:
        fit = [r for r in Hd if r["cluster"] != c]
        f = scorer(cfg, fit)
        if mode == "0.5":
            th = th50 = 0.5
        else:
            th = q99([f(r["id"]) for r in fit])
            th50 = q99([f(r["id"]) for r in H50 if r["cluster"] != c]) if mode in ("len", "deploy") else th
            th100 = q99([f(r["id"]) for r in H100 if r["cluster"] != c]) if mode == "deploy" else th
        TH = {"full": th, "100": th100 if mode == "deploy" else th, "50": th50 if mode == "deploy" else th}
        pick = (lambda r: TH[BK(r["n_words"])]) if mode == "deploy" else (lambda r: th)
        for r in Hd:
            if r["cluster"] == c:
                hit["fpr"].append(f(r["id"]) >= pick(r))
                if r["domain"] == "web":
                    hit["fpr_web"].append(f(r["id"]) >= pick(r))
        for r in H50:
            if r["cluster"] == c:
                hit["fpr50"].append(f(r["id"]) >= th50)
        for r in Md:
            if r["cluster"] == c:
                g = r["generator"]
                key = "tpr_kazllm" if "KazLLM" in g else "tpr_sherkala" if "Sherkala" in g else "tpr_gemma" if g in GM else "tpr_indist"
                hit[key].append(f(r["id"]) >= pick(r))
    out = {k: float(np.mean(v)) for k, v in hit.items()}
    out["n"] = {k: len(v) for k, v in hit.items()}
    return out

ROWS = [("auglen-deploy", "det_auglen", "deploy"), ("combo-deploy", "combo:det_auglen", "deploy"), ("kzlen-deploy", "det_kzlen", "deploy"), ("base05", "det_base", "0.5"), ("base", "det_base", "cal"), ("aug", "det_aug", "cal"), ("auglen", "det_auglen", "cal"),
        ("auglen-lenthr", "det_auglen", "len"), ("combo", "combo:det_auglen", "len"),
        ("kznoaug", "det_kznoaug", "cal"), ("kz", "det_kz", "cal"), ("kzlen", "det_kzlen", "len")]
OUT = {"n_dev_humans": len(Hd), "n_dev_humans_50w": len(H50), "n_dev_machine": len(Md), "clusters": len(CL), "rows": {}}
for name, cfg, mode in ROWS:
    OUT["rows"][name] = run(cfg, mode)
OUT["combos"] = {a: run("combo:" + a, "len") for a in ("det_base", "det_aug", "det_auglen")}
OUT["combos_deploy"] = {a: run("combo:" + a, "deploy") for a in ("det_base", "det_aug", "det_auglen")}
OUT["auroc_dev"] = {t: float(roc_auc_score([1] * sum(1 for r in Md if r["generator"] not in HELD) + [0] * len(Hd),
                                            [S[t][r["id"]] for r in Md if r["generator"] not in HELD] + [S[t][r["id"]] for r in Hd]))
                    for t in ("det_base", "det_aug", "det_auglen", "det_kznoaug", "det_kz", "det_kzlen", "bino")}
os.makedirs("results", exist_ok=True)
json.dump(OUT, open("results/dev_replication_v2.json", "w"), indent=1)
for k, v in list(OUT["rows"].items()) + [("COMBO " + k, v) for k, v in OUT["combos"].items()] + [("COMBOdep " + k, v) for k, v in OUT["combos_deploy"].items()]:
    print(f"{k:16s}", " ".join(f"{m}={100 * v[m]:.1f}" for m in ("fpr", "fpr_web", "fpr50", "tpr_indist", "tpr_gemma", "tpr_kazllm", "tpr_sherkala") if m in v))
print(OUT["n"] if "n" in OUT else "", {k: OUT[k] for k in ("n_dev_humans", "n_dev_humans_50w", "n_dev_machine")})
