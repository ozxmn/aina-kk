import json, glob, sys, re
sys.path.insert(0, "scripts")
import fasttext
from datasketch import MinHash, MinHashLSH
from textnorm import normalize, homoglyph_count, kz_ratio, scrub_pii
MODEL = glob.glob("data/raw/hf/models--cis-lmu--glotlid/snapshots/*/model.bin")[0]
lid = fasttext.load_model(MODEL)
def shingles(t, k=5):
    w = re.findall(r"\w+", t.lower()); return {" ".join(w[i:i+k]) for i in range(max(1, len(w)-k+1))}
rows = []
for fn in ["data/raw/wiki_kk.jsonl", "data/raw/news_kk.jsonl", "data/raw/cc100_kk.jsonl"]:
    rows += [json.loads(l) for l in open(fn, encoding="utf-8")]
lsh = MinHashLSH(threshold=0.8, num_perm=128)
kept, drop = [], {"lid": 0, "kz": 0, "dup": 0}
for r in rows:
    r["text"] = scrub_pii(r["text"])
    lab, p = lid.predict(normalize(r["text"]).replace("\n", " "), k=1)
    r["glotlid"], r["glotlid_p"] = lab[0].replace("__label__", ""), float(p[0])
    if r["glotlid"] != "kaz_Cyrl" or r["glotlid_p"] < 0.9: drop["lid"] += 1; continue
    r["kz_ratio"] = round(kz_ratio(r["text"]), 4)
    if r["kz_ratio"] < 0.02: drop["kz"] += 1; continue
    r["homoglyphs"] = homoglyph_count(r["text"]); r["n_words"] = len(r["text"].split())
    m = MinHash(num_perm=128)
    for s in shingles(r["text"]): m.update(s.encode())
    if lsh.query(m): drop["dup"] += 1; continue
    lsh.insert(r["id"], m); kept.append(r)
with open("data/human/human_pool.jsonl", "w", encoding="utf-8") as f:
    for r in kept: f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print("in", len(rows), "kept", len(kept), drop, Counter(r["domain"] for r in kept))
print("passages with >=1 Latin homoglyph:", Counter(r["domain"] for r in kept if r["homoglyphs"] > 0))
