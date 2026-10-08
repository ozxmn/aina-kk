import sys, json, re, hashlib, random
import pyarrow.parquet as pq
from urllib.parse import urlparse
src, out = sys.argv[1], sys.argv[2]
PER = 40
EDU_HOST = re.compile(r"bilim|referat|stud|kursik|emirsaba|engime|kazref|kzref|zharar|oqu|mektep|ustaz|shygarma|infourok|massaget|bigox|dereksiz|kazaksha|ulagat|energine|yaneuch|kitap|kopilka|uchitel|school|edu", re.I)
KZ = set("әғқңөұүһі")
GENRE = re.compile(r"\bшығарма\b|\bэссе\b|\bбаяндама\b|\bреферат\b|курстық жұмыс|дипломдық жұмыс", re.I)
GENRE_URL = re.compile(r"shygarma|sygarma|shigarma|esse|essay|referat|bayandama|kurstyq|diplom", re.I)
NOT_ESSAY = re.compile(r"мақсаты:|міндеттері:|сабақтың мақсаты|сабақтың тақырыбы|сабақ жоспары|тест|жауаптары|күнтізбелік|оқу бағдарламасы|сынып жетекші|ашық сабақ", re.I)
BAD = re.compile(r"http|www\.|©|telegram|whatsapp|instagram|facebook|оқыңыз:|жазылыңыз|cookie|скачать|жүктеу|бағасы|тг\.", re.I)
NEWS_HOSTS = {"egemen.kz","inform.kz","azattyq.org","baq.kz","abai.kz","tengrinews.kz","qamshy.com","ult.kz","aikyn.kz","turkystan.kz",
              "zhasalash.kz","alashainasy.kz","kazgazeta.kz","kazakhstan.kz","24.kz","nur.kz","adyrna.kz","zhasorken.kz","kazradio.kz",
              "khabar.kz","dalanews.kz","qazaquni.kz","ana-tili.kz","anatili.kz","kaz.zakon.kz","kazpravda.kz","jasalash.kz","qazaqstan.tv",
              "stan.kz","bilimdinews.kz","kznews.kz","sn.kz","syrboyi.kz","mtrk.kz","jylyoi.kz","baribar.kz"}
def passage(text):
    paras = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n")]
    paras = [p for p in paras if len(p.split()) >= 12 and not BAD.search(p)]
    outp = []
    for p in paras:
        if len(" ".join(outp + [p]).split()) > 320:
            if len(" ".join(outp).split()) >= 120: break
            outp = []; continue
        outp.append(p)
        if len(" ".join(outp).split()) >= 160: break
    t = "\n\n".join(outp); n = len(t.split())
    return t if 120 <= n <= 320 else None
def host(u): return urlparse(u).netloc.lower().removeprefix("www.")
pf = pq.ParquetFile(src)
rng = random.Random(7)
res, seen, got = [], set(), {}
for i in range(pf.num_row_groups):
    for r in pf.read_row_group(i, columns=["id", "url", "date", "dump", "text"]).to_pylist():
        if r["date"] >= "2022-11-30": continue
        h = host(r["url"])
        if h in NEWS_HOSTS or got.get(h, 0) >= PER: continue
        head = r["text"][:300]
        if not EDU_HOST.search(h) or not GENRE.search(head) or NOT_ESSAY.search(r["text"][:1500]): continue
        t = passage(r["text"])
        if not t: continue
        letters = [c for c in t.lower() if c.isalpha()]
        if sum(c in KZ for c in letters) / max(1, len(letters)) < 0.015: continue
        k = hashlib.md5(t[:200].encode()).hexdigest()
        if k in seen: continue
        seen.add(k); got[h] = got.get(h, 0) + 1
        res.append({"id": f"essay-{r['id']}", "group": f"essay-{k[:12]}", "domain": "essay", "lang": "kk", "source": h, "url": r["url"],
                    "crawl_date": r["date"], "dump": r["dump"], "fineweb2_id": r["id"], "license": "site copyright (release URL + id only)", "text": t})
rng.shuffle(res)
with open(out, "w", encoding="utf-8") as f:
    for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("passages", len(res), "hosts", len(got))
print(sorted(got.items(), key=lambda x: -x[1])[:25])
