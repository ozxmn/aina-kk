import json, glob, re, sys
sys.path.insert(0, "scripts")
import fasttext
from textnorm import normalize, homoglyph_count, kz_ratio, drop_md_headers, strip_title, to_sentence_end, mixed_script_token
lid = fasttext.load_model(glob.glob("data/raw/hf/models--cis-lmu--glotlid/snapshots/*/model.bin")[0])
LABEL = {"C1": "M", "C2": "M", "C3": "HP", "C4": "MT", "C5": "MT", "C6": "MH", "C7": "M"}
META_HEAD = re.compile(r"^(міне|әрине|жақсы|сұрауыңыз бойынша|төменде|here is|sure|certainly|мақала:|мәтін:)[^\n]{0,120}[:：]?\s*$", re.I)
META_TAIL = re.compile(r"(тағы (бір )?(нәрсе|көмек)|сұрақтарыңыз болса|қосымша ақпарат қажет болса|let me know)[^\n]*$", re.I)
PROMPT_LEAK = re.compile(r"\d+\S*\s+(аса\s+)?сөз(бен|ден)\s+(айтқанда|аспайтын|тұратын мәтін)|сөз саны|word count", re.I)
REFUSAL = re.compile(r"(кешіріңіз, (мен )?(бұл|бұны|оны) (жаза|орындай)|i can'?t|i cannot|as an ai|мен жасанды интеллект ретінде)", re.I)
def trim_words(t, hi=320):
    if len(t.split()) <= hi: return t
    sents = re.split(r"(?<=[.!?…])\s+", t); out = []
    for s in sents:
        if len(" ".join(out + [s]).split()) > hi: break
        out.append(s)
    return " ".join(out)
def rep_ratio(t):
    w = t.lower().split(); g = [" ".join(w[i:i+3]) for i in range(len(w) - 2)]
    return len(set(g)) / max(1, len(g))
def clean(rec):
    t = (rec.get("text") or "").strip(); notes = []
    lines = t.split("\n")
    while lines and (not lines[0].strip() or META_HEAD.match(normalize(lines[0]).strip())):
        if lines[0].strip(): notes.append("meta_head")
        lines = lines[1:]
    while lines and (not lines[-1].strip() or META_TAIL.search(lines[-1])):
        if lines[-1].strip(): notes.append("meta_tail")
        lines = lines[:-1]
    t = "\n".join(lines).strip()
    t2 = drop_md_headers(t)
    if t2 != t: notes.append("md_headers")
    t, st = strip_title(t2)
    if st: notes.append("title_stripped")
    if rec["condition"] == "C2" and rec.get("human_prefix"):
        pf = rec["human_prefix"]
        if t[:40] and pf.startswith(t[:40]): t = t[len(pf):].lstrip(); notes.append("prefix_repeat")
    fin = str((rec.get("meta") or {}).get("finish", "")).lower()
    t2 = to_sentence_end(t)
    if t2 != t: notes.append("cut_fragment" if fin not in ("length", "max_tokens") else "truncated_to_sentence")
    t = t2
    t2 = trim_words(t)
    if t2 != t: notes.append("trimmed_320")
    return t2, notes
def audit_drops():
    out = {}
    for fn in ("results/llm_audit.jsonl", "results/human_audit.jsonl"):
        try:
            for l in open(fn, encoding="utf-8"):
                r = json.loads(l)
                if r.get("kind") == "machine" and r.get("ok") is False and r.get("reason") != "looks_mt": out[r["id"]] = r.get("reason")
        except FileNotFoundError: pass
    return out
def main():
    AUD = audit_drops()
    files = sorted(glob.glob("data/machine/*.out.jsonl")) + sorted(glob.glob("kaggle/out*/*.out.jsonl"))
    seen, out, drops = set(), [], {}
    for fn in files:
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l)
            if r.get("status") != 200 or not r.get("text") or r["task_id"] in seen: continue
            seen.add(r["task_id"])
            t, notes = clean(r)
            reason = None
            nw = len(t.split())
            if nw < 60: reason = "too_short"
            elif REFUSAL.search(t[:300]): reason = "refusal"
            elif rep_ratio(t) < 0.55: reason = "repetition"
            elif PROMPT_LEAK.search(t): reason = "prompt_leak"
            elif mixed_script_token(t): reason = "mixed_script"
            else:
                lab, p = lid.predict(normalize(t).replace("\n", " "), k=1)
                if lab[0] != "__label__kaz_Cyrl" or p[0] < 0.9: reason = f"lid:{lab[0].replace('__label__','')}"
            gen = r["generator"]
            if reason:
                drops.setdefault(gen, {}).setdefault(reason.split(":")[0], 0); drops[gen][reason.split(":")[0]] += 1; continue
            base = {k: r.get(k) for k in ("task_id", "generator", "condition", "source_id", "group", "domain", "split", "cluster", "temperature", "ts")}
            base.update({"label": LABEL[r["condition"]], "text": t, "postproc": notes, "n_words": nw, "homoglyphs": homoglyph_count(t),
                         "kz_ratio": round(kz_ratio(t), 4), "model_version": (r.get("meta") or {}).get("modelVersion") or (r.get("meta") or {}).get("model")})
            out.append({**base, "id": "m-" + str(abs(hash(r["task_id"])) % 10**12)})
            if r["condition"] == "C2" and r.get("human_prefix"):
                mix = r["human_prefix"].strip() + " " + t
                out.append({**base, "id": "x-" + str(abs(hash(r["task_id"])) % 10**12), "label": "MIX", "text": trim_words(mix, 360),
                            "boundary_char": len(r["human_prefix"].strip()) + 1, "n_words": len(mix.split())})
    import hashlib
    for o in out: o["id"] = o["id"][:2] + hashlib.md5(o["task_id"].encode()).hexdigest()[:12]
    flagged = {o["task_id"] for o in out if o["id"] in AUD}
    for o in out:
        if o["task_id"] in flagged and o["id"].startswith("m-"):
            g = o["generator"]; drops.setdefault(g, {}).setdefault("audit_flag", 0); drops[g]["audit_flag"] += 1
    out = [o for o in out if o["task_id"] not in flagged]
    with open("data/machine/machine_pool.jsonl", "w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    from collections import Counter
    print("kept", len(out), Counter((o["generator"].split("/")[-1].split(":")[-1], o["label"]) for o in out))
    print("drops", drops)
    json.dump({"kept": len(out), "drops": drops}, open("results/qc_machine_log.json", "w"), indent=1)
if __name__ == "__main__": main()
