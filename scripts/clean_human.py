import json, os, re, sys
from collections import Counter
sys.path.insert(0, "scripts")
from textnorm import strip_title, to_sentence_end, mixed_script_token

JUNK = {
    "markup": re.compile(r"</?noinclude|\{\{|\}\}|\[\[|\]\]"),
    "download_banner": re.compile(r"Презентация на тему|Скачать (бесплатно|презентацию)|без регистрации"),
    "breadcrumb": re.compile(r"(Home|Басты бет|Главная)\s*»"),
    "related_links": re.compile(r"Осы категория бойынша|Ұқсас жаңалықтар|Тағы оқыңыз|Пікір қалдыр|Бөлісу:|Тегтер:|Теги:"),
}
GLUE = re.compile(r"[а-яәіңғүұқөһ.!?\"»)]\d{1,2}[А-ЯӘІҢҒҮҰҚӨҺA-Z\"«]")

def flagged_ids():
    ids = {}
    for fn, who in (("results/llm_audit.jsonl", "ai_check"), ("results/human_audit.jsonl", "human_audit")):
        if not os.path.exists(fn): continue
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l)
            if r.get("kind") == "human" and r.get("ok") is False: ids.setdefault(r["id"], []).append(f"{who}:{r.get('reason')}")
    return ids

def main():
    H = [json.loads(l) for l in open("data/human/human.jsonl", encoding="utf-8")]
    flags = flagged_ids()
    out, drops, edits = [], Counter(), Counter()
    for h in H:
        t = h["text"]; why = None
        for k, p in JUNK.items():
            if p.search(t): why = k; break
        if not why and len(GLUE.findall(t)) >= 2: why = "headline_glue"
        if not why and mixed_script_token(t): why = "mixed_script"
        if not why and h["id"] in flags: why = "audit_flag"
        if why:
            drops[(why, h["domain"])] += 1; continue
        notes = []
        t, st = strip_title(t)
        if st: notes.append("title_stripped")
        t2 = to_sentence_end(t)
        if t2 != t: notes.append("cut_fragment")
        t = t2
        if len(t.split()) < 60:
            drops[("too_short_after_trim", h["domain"])] += 1; continue
        for n in notes: edits[(n, h["domain"])] += 1
        out.append({**h, "text": t, "n_words": len(t.split()), "postproc": notes})
    with open("data/human/human_clean.jsonl", "w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    log = {"n_in": len(H), "n_out": len(out), "drops": {f"{a}|{b}": c for (a, b), c in sorted(drops.items())},
           "edits": {f"{a}|{b}": c for (a, b), c in sorted(edits.items())}, "audit_flagged": len(flags)}
    json.dump(log, open("results/clean_human_log.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(log, indent=1, ensure_ascii=False))
    print("by split/domain:", Counter((o["split"], o["domain"]) for o in out))

if __name__ == "__main__": main()
