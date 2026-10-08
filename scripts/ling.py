import json, re, numpy as np
from scipy.stats import mannwhitneyu
R = [json.loads(l) for l in open("data/eval/eval.jsonl", encoding="utf-8")]
H = [r for r in R if r["label"] == "H" and r["variant"] == "orig"]
M = [r for r in R if r["label"] == "M" and r["variant"] == "orig"]
KZ = set("әғқңөұүһі")
def words(t): return re.findall(r"[^\W\d_]+", t.lower())
def mattr(w, k=50):
    if len(w) < k: return len(set(w)) / max(1, len(w))
    return float(np.mean([len(set(w[i:i+k])) / k for i in range(len(w) - k + 1)]))
def feats(t):
    w = words(t); sents = [s for s in re.split(r"(?<=[.!?…])\s+", t) if s.strip()]
    sl = [len(words(s)) for s in sents] or [0]
    n = max(1, len(w))
    return {"MATTR(50)": mattr(w), "mean sentence length": float(np.mean(sl)), "sentence length SD": float(np.std(sl)),
            "'болып табылады' per 1k words": 1000 * len(re.findall(r"болып табыла", t.lower())) / n,
            "Kazakh-specific letter share": sum(c in KZ for c in t.lower() if c.isalpha()) / max(1, sum(c.isalpha() for c in t)),
            "mean word length": float(np.mean([len(x) for x in w])) if w else 0,
            "commas per sentence": t.count(",") / max(1, len(sents)),
            "-ған/-ген participles per 100 words": 100 * sum(bool(re.search(r"(ған|ген|қан|кен)(ы|і|дар|дер|лар|лер)?$", x)) for x in w) / n,
            "'және' per 100 words": 100 * w.count("және") / n}
def cliffs(a, b):
    a = np.asarray(a); b = np.asarray(b); return float(((a[:, None] > b[None]).mean() - (a[:, None] < b[None]).mean()))
FH = [feats(r["text"]) for r in H]; FM = [feats(r["text"]) for r in M]
rows = []
for k in FH[0]:
    a = [f[k] for f in FH]; b = [f[k] for f in FM]
    p = mannwhitneyu(b, a).pvalue
    rows.append([k, float(np.median(a)), float(np.median(b)), cliffs(b, a), p])
ps = np.array([r[4] for r in rows]); order = np.argsort(ps); m = len(ps); q = np.empty(m)
prev = 1.0
for rank, i in enumerate(order[::-1]):
    j = m - rank; prev = min(prev, ps[i] * m / j); q[i] = prev
out = [{"feature": r[0], "median_human": r[1], "median_machine": r[2], "cliffs_delta_machine_vs_human": r[3], "p": r[4], "q_bh": float(q[i])} for i, r in enumerate(rows)]
json.dump({"n_human": len(H), "n_machine": len(M), "features": out}, open("results/linguistic.json", "w"), indent=1, ensure_ascii=False)
for o in out: print(f"{o['feature']:40s} H {o['median_human']:.3f}  M {o['median_machine']:.3f}  δ {o['cliffs_delta_machine_vs_human']:+.2f}  q {o['q_bh']:.1e}")
