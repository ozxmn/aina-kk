import sys, json, re, hashlib, random
import pyarrow.parquet as pq
from urllib.parse import urlparse
src, out = sys.argv[1], sys.argv[2]
DOMAINS = ["egemen.kz","inform.kz","azattyq.org","baq.kz","abai.kz","tengrinews.kz","massaget.kz","qamshy.com","ult.kz","aikyn.kz",
 "turkystan.kz","zhasalash.kz","alashainasy.kz","kazgazeta.kz","kazakhstan.kz","24.kz","nur.kz","adyrna.kz","zhasorken.kz","kazradio.kz",
 "khabar.kz","dalanews.kz","qazaquni.kz","ana-tili.kz","anatili.kz","kaz.zakon.kz","kazpravda.kz","jasalash.kz","qazaqstan.tv","stan.kz","bilimdinews.kz"]
PER = 60
KZ = set("әғқңөұүһі")
BAD = re.compile(r"http|www\.|©|telegram|whatsapp|instagram|facebook|оқыңыз:|жазылыңыз|cookie|материалды көшіріп", re.I)
def passage(text):
    paras = [re.sub(r"\s+"," ",p).strip() for p in text.split("\n")]
    paras = [p for p in paras if len(p.split()) >= 12 and not BAD.search(p)]
    out = []
    for p in paras:
        if len(" ".join(out + [p]).split()) > 320:
            if len(" ".join(out).split()) >= 120: break
            out = []; continue
        out.append(p)
        if len(" ".join(out).split()) >= 160: break
    t = "\n\n".join(out); n = len(t.split())
    return t if 120 <= n <= 320 else None
def dom_of(u):
    h = urlparse(u).netloc.lower().removeprefix("www.")
    for d in DOMAINS:
        if h == d or h.endswith("." + d): return d
    return None
pf = pq.ParquetFile(src)
cands = {}
for i in range(pf.num_row_groups):
    t = pf.read_row_group(i, columns=["id","url","date","dump"]).to_pylist()
    for j, r in enumerate(t):
        d = dom_of(r["url"])
        if d and r["date"] < "2022-11-30":
            cands.setdefault(d, []).append((i, j))
print({k: len(v) for k, v in cands.items()}, flush=True)
rng = random.Random(3)
pick = {}
for d, lst in cands.items():
    rng.shuffle(lst); pick[d] = lst[: PER * 4]
need = {}
for d, lst in pick.items():
    for i, j in lst: need.setdefault(i, []).append((j, d))
res, seen, got = [], set(), {d: 0 for d in pick}
for i, lst in sorted(need.items()):
    tbl = pf.read_row_group(i, columns=["id","url","date","dump","text"]).to_pylist()
    for j, d in lst:
        if got[d] >= PER: continue
        r = tbl[j]; t = passage(r["text"])
        if not t: continue
        letters = [c for c in t.lower() if c.isalpha()]
        if sum(c in KZ for c in letters) / max(1, len(letters)) < 0.015: continue
        h = hashlib.md5(t[:200].encode()).hexdigest()
        if h in seen: continue
        seen.add(h); got[d] += 1
        res.append({"id": f"news-{r['id']}", "group": f"news-{h[:12]}", "domain": "news", "lang": "kk", "source": d, "url": r["url"],
                    "crawl_date": r["date"], "dump": r["dump"], "fineweb2_id": r["id"], "license": "publisher copyright (release URL + id only)", "text": t})
with open(out, "w", encoding="utf-8") as f:
    for x in res: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(got, "kept", len(res))
