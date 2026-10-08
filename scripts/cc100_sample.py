import sys, json, random, re, hashlib
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
KZ = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
BAD = re.compile(r"wikipedia|бетінен алынған|http|www\.|©|cookie|\.com|@", re.I)

def ok_para(p):
    w = p.split()
    if len(w) < 12: return False
    letters = [c for c in p if c.isalpha()]
    if not letters: return False
    cyr = sum('Ѐ' <= c <= 'ӿ' for c in letters) / len(letters)
    kz = sum(c in KZ for c in letters) / len(letters)
    digits = sum(c.isdigit() for c in p) / len(p)
    return cyr > 0.9 and kz > 0.015 and digits < 0.08 and not BAD.search(p) and p.rstrip()[-1:] in ".!?»\"…"

docs, cur = [], []
with open(src, encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            if cur: docs.append(cur); cur = []
        else: cur.append(line)
random.Random(13).shuffle(docs)
res, seen = [], set()
for i, d in enumerate(docs):
    if any(BAD.search(l) for l in d): continue
    paras = [p for p in d if ok_para(p)]
    out_p = []
    for p in paras:
        if len(" ".join(out_p + [p]).split()) > 320:
            if len(" ".join(out_p).split()) >= 120: break
            out_p = []; continue
        out_p.append(p)
        if len(" ".join(out_p).split()) >= 160: break
    text = "\n\n".join(out_p); nw = len(text.split())
    if not (120 <= nw <= 320): continue
    h = hashlib.md5(text[:200].encode()).hexdigest()
    if h in seen: continue
    seen.add(h)
    res.append({"id": f"cc100-{h[:12]}", "group": f"cc100-{h[:12]}", "domain": "web", "lang": "kk", "source": "CC100-kk (2018-19 crawl)",
                "license": "CC100 / Common Crawl terms (release ids only)", "text": text})
    if len(res) >= n: break
with open(out, "w", encoding="utf-8") as f:
    for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("docs", len(docs), "kept", len(res))
