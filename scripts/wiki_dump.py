import bz2, sys, json, random, re
import xml.etree.ElementTree as ET
import mwparserfromhell as mwp
src, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = random.Random(7)
KZ = set("әғқңөұүһі")
def clean(raw):
    raw = re.sub(r"(?s)<ref[^>]*?/>|<ref.*?</ref>|<!--.*?-->|\{\|.*?\|\}", " ", raw)
    t = mwp.parse(raw).strip_code(normalize=True, collapse=True)
    paras = []
    for p in re.split(r"\n\s*\n|\n", t):
        p = re.sub(r"\s+", " ", p).strip()
        if len(p.split()) < 12 or p.startswith(("Санат:", "Сурет:", "Файл:", "thumb", "*", "|", "!")): continue
        letters = [c for c in p.lower() if c.isalpha()]
        if not letters or sum('Ѐ' <= c <= 'ӿ' for c in letters) / len(letters) < 0.85: continue
        if sum(c in KZ for c in letters) / len(letters) < 0.015: continue
        paras.append(re.sub(r"\(\s*[;,]?\s*\)", "", p))
    return paras
def passage(paras, lo=120, hi=320):
    out = []
    for p in paras:
        if len(" ".join(out + [p]).split()) > hi:
            if len(" ".join(out).split()) >= lo: break
            out = []
            if len(p.split()) > hi: continue
        out.append(p)
        if len(" ".join(out).split()) >= 160: break
    t = "\n\n".join(out); n = len(t.split())
    return t if lo <= n <= hi else None
ns = "{http://www.mediawiki.org/xml/export-0.10/}"
res, seen_pages, parsed = [], 0, 0
with bz2.open(src, "rb") as f:
    for ev, el in ET.iterparse(f, events=("end",)):
        if el.tag != ns + "page": continue
        seen_pages += 1
        if el.findtext(ns + "ns") == "0" and el.find(ns + "redirect") is None:
            rev = el.find(ns + "revision"); raw = rev.findtext(ns + "text") or ""
            if len(raw) >= 4000 and not raw.lstrip().lower().startswith("#") and rng.random() < 0.2:
                parsed += 1
                t = passage(clean(raw))
                if t:
                    pid = el.findtext(ns + "id"); title = el.findtext(ns + "title")
                    res.append({"id": f"wiki-kk-{pid}", "group": f"wiki-kk-{pid}", "domain": "encyclopedic", "lang": "kk",
                                "source": "kk.wikipedia dump 2022-05-01", "title": title, "revid": rev.findtext(ns + "id"),
                                "rev_timestamp": rev.findtext(ns + "timestamp"), "license": "CC BY-SA 4.0", "text": t})
        el.clear()
        if seen_pages % 50000 == 0: print("pages", seen_pages, "parsed", parsed, "kept", len(res), flush=True)
rng.shuffle(res)
with open(out, "w", encoding="utf-8") as g:
    for r in res[:N]: g.write(json.dumps(r, ensure_ascii=False) + "\n")
print("done pages", seen_pages, "parsed", parsed, "kept", len(res), "written", min(N, len(res)))
