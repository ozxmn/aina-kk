import json, random, re, hashlib
docs, cur = [], []
for line in open("data/raw/cc100_ru_part.txt", encoding="utf-8", errors="ignore"):
    line = line.strip()
    if not line:
        if cur: docs.append(cur); cur = []
    else: cur.append(line)
rng = random.Random(5); rng.shuffle(docs)
BAD = re.compile(r"http|www\.|©|cookie|\.ru|\.com|@|скачать|казино|купить", re.I)
res = []
for d in docs:
    if any(BAD.search(l) for l in d): continue
    paras = [p for p in d if len(p.split()) >= 12 and p[-1:] in ".!?»"]
    out = []
    for p in paras:
        if len(" ".join(out + [p]).split()) > 260:
            if len(" ".join(out).split()) >= 110: break
            out = []; continue
        out.append(p)
        if len(" ".join(out).split()) >= 140: break
    t = "\n\n".join(out); n = len(t.split())
    if 110 <= n <= 260:
        h = hashlib.md5(t[:200].encode()).hexdigest()
        res.append({"id": f"ru-{h[:12]}", "group": f"ru-{h[:12]}", "domain": "web-ru", "lang": "ru", "split": rng.choice(["train", "train", "train", "test", "test"]),
                    "source": "CC100-ru (2018-19 crawl)", "text": t})
    if len(res) >= 600: break
with open("data/human/ru_pool.jsonl", "w", encoding="utf-8") as f:
    for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("ru passages", len(res))
