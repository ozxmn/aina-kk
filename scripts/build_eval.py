import json, random, os, sys, re
sys.path.insert(0, "scripts")
from textnorm import normalize, fold_homoglyphs, CYR2LAT, homoglyph_count
from translit2021 import to_latin, to_cyrillic
rng = random.Random(77)
HELDOUT = ("gemma-4", "Sherkala", "KazLLM")
import os
HFILES = ["data/human/human_clean.jsonl", "data/human/essay_clean.jsonl"]
H = [json.loads(l) for fn in HFILES if os.path.exists(fn) for l in open(fn, encoding="utf-8")]
M = [json.loads(l) for l in open("data/machine/machine_pool.jsonl", encoding="utf-8")]
rows = []
for h in H:
    rows.append({"id": h["id"], "base_id": h["id"], "variant": "orig", "label": "H", "split": h["split"], "domain": h["domain"],
                 "generator": "human", "condition": "human", "heldout": False, "group": h["group"], "cluster": h.get("cluster"), "src_split": h["split"],
                 "homoglyphs": h.get("homoglyphs", 0), "raw": h["text"]})
for m in M:
    g = m["generator"].split(":", 1)[1].split("/")[-1]
    held = any(x in g for x in HELDOUT)
    split = m["split"] if m["label"] == "M" and not held else "test"
    if m["label"] == "M" and held: split = "test"
    rows.append({"id": m["id"], "base_id": m["id"], "variant": "orig", "label": m["label"], "split": split, "domain": m["domain"],
                 "generator": g, "condition": m["condition"], "heldout": held, "group": m["group"], "cluster": m.get("cluster"), "src_split": m["split"],
                 "homoglyphs": m.get("homoglyphs", 0), "raw": m["text"]})
test_h = [r for r in rows if r["label"] == "H" and r["split"] == "test"]
test_m = [r for r in rows if r["label"] == "M" and r["split"] == "test" and not r["heldout"]]
def var(r, name, text):
    v = dict(r); v.update({"id": f"{r['id']}~{name}", "variant": name, "raw": text}); return v
def homoglyph_attack(t, frac, rr):
    out = []
    for c in t:
        if c in CYR2LAT and rr.random() < frac: out.append(CYR2LAT[c])
        else: out.append(c)
    return "".join(out)
import hashlib
def u(i, salt): return int(hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()[:12], 16) / 16 ** 12
extra = []
PH_LAT, PH_TR = 500 / 1530, 400 / 1530
for r in [x for x in test_m if u(x["id"], "hg") < 0.34]:
    hg = homoglyph_attack(r["raw"], 0.10, random.Random(f"hg:{r['id']}"))
    extra.append(var(r, "homoglyph10", hg)); extra.append(var(r, "homoglyph10fold", fold_homoglyphs(hg)))
for r in [x for x in test_h if u(x["id"], "lat") < PH_LAT] + [x for x in test_m if u(x["id"], "lat") < 0.34]:
    lat = to_latin(r["raw"])
    extra.append(var(r, "latin2021", lat)); extra.append(var(r, "latin2021canon", to_cyrillic(lat)))
for r in [x for x in test_h if u(x["id"], "tr") < PH_TR] + [x for x in test_m if u(x["id"], "tr") < 0.30]:
    w = r["raw"].split()
    extra.append(var(r, "trunc100", " ".join(w[:100])))
    extra.append(var(r, "trunc50", " ".join(w[:50])))
for r in [x for x in rows if x["label"] == "H" and x["homoglyphs"] > 0]:
    extra.append(var(r, "folded", fold_homoglyphs(r["raw"])))
train_cl = sorted({r["cluster"] for r in rows if r["label"] == "H" and r["src_split"] == "train" and r["cluster"] is not None})
DEV_CL = set(random.Random(5).sample(train_cl, max(1, len(train_cl) // 5)))
for r in [x for x in rows if x["label"] == "H" and x["src_split"] == "train" and x["cluster"] in DEV_CL]:
    w = r["raw"].split()
    extra.append(var(r, "devtrunc100", " ".join(w[:100])))
    extra.append(var(r, "devtrunc50", " ".join(w[:50])))
rows += extra
os.makedirs("data/eval", exist_ok=True)
with open("data/eval/eval_raw.jsonl", "w", encoding="utf-8") as fr, open("data/eval/eval.jsonl", "w", encoding="utf-8") as fn:
    for r in rows:
        fr.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = {k: v for k, v in r.items() if k != "raw"}; n["text"] = normalize(r["raw"])
        fn.write(json.dumps(n, ensure_ascii=False) + "\n")
from collections import Counter
print(len(rows), Counter((r["label"], r["split"], r["variant"]) for r in rows).most_common(30))
print("generators", Counter(r["generator"] for r in rows if r["variant"] == "orig"))
