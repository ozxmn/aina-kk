import json, math, os, random, subprocess, sys
from collections import defaultdict
import numpy as np

SCORES, META, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTDIR, exist_ok=True)
R = [json.loads(l) for l in open(SCORES, encoding="utf-8")]
S = {r["id"]: r for r in R}
D = os.path.join(OUTDIR, "scores")
os.makedirs(D, exist_ok=True)
with open(os.path.join(D, "det_auglen_s1.jsonl"), "w") as f:
    for r in R:
        f.write(json.dumps({"id": r["id"], "p_machine": r["classifier"]}) + "\n")
with open(os.path.join(D, "zs_gemma4-e2b.jsonl"), "w") as f:
    for r in R:
        f.write(json.dumps({"id": r["id"], "binoculars": r["binoculars"], "obs_fdg": float("nan"), "perf_fdg": float("nan")}) + "\n")
json.dump(json.load(open(META)), open(os.path.join(D, "det_meta.json"), "w"))
AN = os.path.join(OUTDIR, "analysis_release.json")
subprocess.run([sys.executable, "scripts/analyze_det.py", D, AN], check=True, stdout=subprocess.DEVNULL)
C = json.load(open(AN))["configs"]["combo_auglen+bino"]
E = {r["id"]: r for r in map(json.loads, open("data/eval/eval.jsonl", encoding="utf-8"))}
DEV = set(json.load(open(META))["dev_clusters"])
hdev = [i for i, r in E.items() if r["label"] == "H" and r["variant"] == "orig" and r["src_split"] == "train" and r["cluster"] in DEV and i in S]
cal = {
    "classifier": "mean P(machine) of three mDeBERTa-v3-base models (seeds 1-3), trained on AINA-KK with script augmentation and length matching",
    "zero_shot": "Binoculars score with google/gemma-4-E2B (observer) and google/gemma-4-E2B-it (performer), first 512 tokens; lower = more machine-like",
    "rule": "each score is mapped to its empirical CDF on the development humans; the text is flagged when the larger of the two values reaches the threshold for its length",
    "dev_human_classifier": [S[i]["classifier"] for i in hdev],
    "dev_human_binoculars": [S[i]["binoculars"] for i in hdev],
    "length_buckets": [["full", 110], ["100", 100], ["50", 50]],
    "thresholds": {"full": C["thr_dev"], "100": C["short"]["100"]["thr_devlen"], "50": C["short"]["50"]["thr_devlen"]},
}

sys.path.insert(0, "detector")
import aina_kk_det as K

class _Det(K.AinaKKDet):
    def __init__(self, cal):
        self.cal = dict(cal, test_rates={})
        self._ref_c = np.sort(np.array(cal["dev_human_classifier"], dtype=float))
        self._ref_b = np.sort(-np.array(cal["dev_human_binoculars"], dtype=float))

det = _Det(cal)
ref_c = np.sort([S[i]["classifier"] for i in hdev])
ref_b = np.sort([-S[i]["binoculars"] for i in hdev])
mism = 0
for i, r in S.items():
    ours = det.decide("w " * 200, r["classifier"], r["binoculars"])["combined"]
    theirs = max(float(np.searchsorted(ref_c, r["classifier"], side="right")) / len(ref_c),
                 float(np.searchsorted(ref_b, -r["binoculars"], side="right")) / len(ref_b))
    mism += abs(ours - theirs) > 1e-12
assert mism == 0, mism
forced = {}
for v, b, n in (("orig", "full", 200), ("trunc100", "100", 100), ("trunc50", "50", 50)):
    hs = [i for i, e in E.items() if e["label"] == "H" and e["variant"] == v and e["split"] == "test" and i in S]
    forced[b] = float(np.mean([det.decide("w " * n, S[i]["classifier"], S[i]["binoculars"])["machine"] for i in hs]))
assert abs(forced["full"] - C["fpr@dev"]) < 1e-9 and abs(forced["100"] - C["short"]["100"]["fpr@devlen"]) < 1e-9 and abs(forced["50"] - C["short"]["50"]["fpr@devlen"]) < 1e-9

def flag(i):
    r = S[i]
    return det.decide("w " * r["n_words"], r["classifier"], r["binoculars"])["machine"]

def rate(ids):
    v = [flag(i) for i in ids if i in S]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)) if v else None), len(v)

def boot(ids, B=1000, seed=0):
    g = defaultdict(list)
    for i in ids:
        if i in S and flag(i) is not None:
            g[E[i]["group"]].append(flag(i))
    keys = list(g)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        xs = [x for k in (rng.choice(keys) for _ in keys) for x in g[k]]
        out.append(np.mean(xs))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

def sel(f):
    return [i for i, e in E.items() if f(e) and i in S]

H_te = sel(lambda e: e["label"] == "H" and e["variant"] == "orig" and e["split"] == "test")
M_id = sel(lambda e: e["label"] == "M" and e["variant"] == "orig" and e["split"] == "test" and not e["heldout"])
M_ho = sel(lambda e: e["label"] == "M" and e["variant"] == "orig" and e["heldout"] and e["src_split"] == "test")
V = lambda lab, var: sel(lambda e: e["label"] == lab and e["variant"] == var)
rows = []

def add(name, ids, ci=False, kind="flagged"):
    r, n = rate(ids)
    rows.append({"name": name, "rate": r, "n": n, "ci": boot(ids) if ci else None, "kind": kind})

add("Human passages, all domains", H_te, True, "false positives")
for d in sorted({E[i]["domain"] for i in H_te}):
    add(f"Human passages, {d}", [i for i in H_te if E[i]["domain"] == d], True, "false positives")
add("Human excerpts, 100 words", V("H", "trunc100"), True, "false positives")
add("Human excerpts, 50 words", V("H", "trunc50"), True, "false positives")
add("Human passages in Latin script", V("H", "latin2021"), True, "false positives")
add("Machine text, training generators", M_id, False, "detected")
add("Machine text, unseen generators", M_ho, False, "detected")
for g in sorted({E[i]["generator"] for i in M_id + M_ho}):
    add(f"Machine text, {g}", [i for i in M_id + M_ho if E[i]["generator"] == g], False, "detected")
add("Machine excerpts, 100 words", V("M", "trunc100"), False, "detected")
add("Machine excerpts, 50 words", V("M", "trunc50"), False, "detected")
add("Machine text in Latin script", V("M", "latin2021"), False, "detected")
add("Machine text with 10% Latin homoglyphs", V("M", "homoglyph10"), False, "detected")
for g, nm in (("HP", "AI-polished human text"), ("MIX", "Human opening + machine continuation"), ("MT", "Machine translation from Russian"), ("MH", "Humanizer rewrites")):
    add(nm, V(g, "orig"), False, "flagged")
by_bucket = {}
for b, rows_h, rows_m in (("full", H_te, M_id), ("100", V("H", "trunc100"), V("M", "trunc100")), ("50", V("H", "trunc50"), V("M", "trunc50"))):
    by_bucket[b] = {"false_positive_rate": rate(rows_h)[0], "false_positive_rate_95ci": boot(rows_h), "detection_rate_training_generators": rate(rows_m)[0]}
cal["test_rates"] = by_bucket
json.dump(cal, open(os.path.join(OUTDIR, "calibration.json"), "w"), indent=1)
json.dump({"rows": rows, "analysis_protocol": {"fpr": C["fpr@dev"], "fpr_ci": C["fpr@dev_ci"], "tpr": C["tpr@dev"], "tpr_heldout": C["tpr_heldout@dev"],
                                                 "fpr50": C["short"]["50"]["fpr@devlen"], "hp": C["flag_gray@dev"]["HP"]},
           "n_dev_humans": len(hdev)}, open(os.path.join(OUTDIR, "rates.json"), "w"), indent=1)
for r in rows:
    print(f"{r['name']:45s} {r['kind']:16s} {100 * r['rate']:6.1f}%  n={r['n']:5d}" + (f"  95% CI {100 * r['ci'][0]:.1f}-{100 * r['ci'][1]:.1f}" if r["ci"] else ""))
print("thresholds", cal["thresholds"])
