import glob, json, re, sys, random
sys.path.insert(0, "scripts")
import fasttext, numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textnorm import normalize, homoglyph_count, kz_ratio, scrub_pii, strip_title, to_sentence_end, mixed_script_token
from clean_human import JUNK, GLUE
lid = fasttext.load_model(glob.glob("data/raw/hf/models--cis-lmu--glotlid/snapshots/*/model.bin")[0])
def shingles(t, k=5):
    w = re.findall(r"\w+", t.lower()); return {" ".join(w[i:i+k]) for i in range(max(1, len(w) - k + 1))}
def mh(t):
    m = MinHash(num_perm=128)
    for s in shingles(t): m.update(s.encode())
    return m
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for i, l in enumerate(open("data/human/human_clean.jsonl", encoding="utf-8")):
    lsh.insert(f"old{i}", mh(json.loads(l)["text"]))
rows, drop = [], {}
def d(k): drop[k] = drop.get(k, 0) + 1
for i, l in enumerate(open("data/raw/essay_kk.jsonl", encoding="utf-8")):
    r = json.loads(l)
    t = scrub_pii(r["text"])
    if any(p.search(t) for p in JUNK.values()) or len(GLUE.findall(t)) >= 2 or mixed_script_token(t): d("junk"); continue
    t, _ = strip_title(t); t = to_sentence_end(t)
    if len(t.split()) < 60: d("short"); continue
    lab, p = lid.predict(normalize(t).replace("\n", " "), k=1)
    if lab[0] != "__label__kaz_Cyrl" or p[0] < 0.9: d("lid"); continue
    if kz_ratio(t) < 0.02: d("kz"); continue
    m = mh(t)
    if lsh.query(m): d("dup"); continue
    lsh.insert(f"new{i}", m)
    r.update({"text": t, "glotlid": "kaz_Cyrl", "glotlid_p": float(p[0]), "kz_ratio": round(kz_ratio(t), 4),
              "homoglyphs": homoglyph_count(t), "n_words": len(t.split())})
    rows.append(r)
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000, sublinear_tf=True).fit_transform([r["text"] for r in rows])
km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(X)
sizes = {c: int((km.labels_ == c).sum()) for c in range(8)}
order = list(range(8)); random.Random(11).shuffle(order)
train, n = set(), 0
for c in order:
    if n + sizes[c] <= 0.62 * len(rows): train.add(c); n += sizes[c]
for r, c in zip(rows, km.labels_):
    r["cluster"] = 100 + int(c); r["split"] = "train" if c in train else "test"
with open("data/human/essay_clean.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print("kept", len(rows), "drops", drop, Counter(r["split"] for r in rows), "hosts", len({r["source"] for r in rows}))
