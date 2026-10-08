import json, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
rows = [json.loads(l) for l in open("data/human/human_pool.jsonl", encoding="utf-8")]
rng = random.Random(42)
news = [r for r in rows if r["domain"] == "news"]; rng.shuffle(news)
rows = [r for r in rows if r["domain"] != "news"] + news[:1100]
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60000, sublinear_tf=True).fit_transform([r["text"] for r in rows])
km = KMeans(n_clusters=60, n_init=4, random_state=0).fit(X)
for r, c in zip(rows, km.labels_): r["cluster"] = int(c)
sizes = np.bincount(km.labels_)
order = list(range(60)); rng.shuffle(order)
test, acc = set(), 0
for c in order:
    if acc + sizes[c] <= 0.4 * len(rows) + 30: test.add(c); acc += sizes[c]
for r in rows: r["split"] = "test" if r["cluster"] in test else "train"
with open("data/human/human.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(len(rows), Counter((r["domain"], r["split"]) for r in rows))
