import argparse, datetime, glob, json, os, re, sys

BATCH = 25
REASONS = ["broken", "boilerplate", "looks_mt", "not_kazakh", "meta_refusal", "other"]
LABEL = {"Human": "HUMAN", "Machine": "MACHINE", "AI-polished human": "AI-POLISHED HUMAN", "Machine-translated": "MACHINE-TRANSLATED"}

INSTRUCTIONS = """You are annotating Kazakh texts for a research dataset. You will receive {n} items. Each item has an ID, a label and a text. The label says what the text is: HUMAN (written by people and published online before 30 November 2022), MACHINE (a language model's answer to a prompt), AI-POLISHED HUMAN (a human text edited by a language model) or MACHINE-TRANSLATED (translated into Kazakh by a machine). The label is correct; do not check it.
For each item, answer one question. For HUMAN items: is this a normal Kazakh text that can be kept as it is? For all other items: is this a normal model answer in Kazakh, without junk?
Answer yes unless one of these problems applies; then answer no and give that reason:
broken: cut off mid-word, letters of Hebrew, Arabic, Greek or other scripts inside words, leftover markup, or garbled text.
boilerplate: website junk such as menus, "Home »", lists of links, "Download", or a reference list.
looks_mt: looks like machine translation, with calques from Russian or incoherent phrases.
not_kazakh: mostly Russian or another language.
meta_refusal: model wrapper, such as a title line at the start, "Міне, мақала:", "Here is...", or a refusal.
other: a problem not in this list.
These are still yes: typos and odd style, factual errors, Latin i, ə or ğ inside Kazakh words, and text that sounds like AI. If unsure, answer yes.
Judge each item on its own. Output one line per item, in the given order, in exactly this format and nothing else:
{{"id": "I1", "valid": true, "reason": null}}
{{"id": "I2", "valid": false, "reason": "broken"}}"""

ITEM = "### {sid} | {label}\n{text}"


def make(items_path, out_dir):
    items = json.load(open(items_path, encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    key = []
    for b in range(0, len(items), BATCH):
        chunk = items[b:b + BATCH]
        parts = [INSTRUCTIONS.format(n=len(chunk)), ""]
        for j, it in enumerate(chunk, b + 1):
            sid = f"I{j}"
            key.append({"sid": sid, "id": it["id"], "kind": it["kind"], "label": LABEL[it["claim"]]})
            parts.append(ITEM.format(sid=sid, label=LABEL[it["claim"]], text=it["text"].strip()))
            parts.append("")
        with open(os.path.join(out_dir, f"batch_{b // BATCH + 1}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(parts).rstrip() + "\n")
    json.dump(key, open(os.path.join(out_dir, "key.json"), "w"), indent=0)
    print(f"{len(items)} items, {len(range(0, len(items), BATCH))} batches in {out_dir}")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def esc(t):
    if t.startswith("{"):
        return "\\texttt{" + t.replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}").replace("_", "\\_") + "}"
    for a, b in (("\\", "\\textbackslash{}"), ("{", "\\{"), ("}", "\\}"), ("_", "\\_"), ("&", "\\&"), ("%", "\\%"),
                 ("#", "\\#"), ("$", "\\$"), ("...", "\\ldots{}")):
        t = t.replace(a, b)
    t = t.replace("\\textbackslash\\{\\}", "\\textbackslash{}")
    t = re.sub(r'"([^"]*)"', r"``\1''", t)
    t = re.sub(r"([\u0400-\u04FF][\u0400-\u04FF\s,:]*[\u0400-\u04FF:])", r"\\kk{\1}", t)
    return t


def score(batch_dir, replies_glob, human_path, out_jsonl, paper_dir, date, model):
    key = json.load(open(os.path.join(batch_dir, "key.json")))
    got = {}
    for fn in sorted(glob.glob(replies_glob)):
        for line in open(fn, encoding="utf-8"):
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            x = json.loads(m.group(0))
            if x["id"] in got:
                sys.exit(f"duplicate judgment for {x['id']}")
            if x["valid"] not in (True, False) or (not x["valid"] and x["reason"] not in REASONS):
                sys.exit(f"bad line: {line.strip()}")
            got[x["id"]] = x
    missing = [k["sid"] for k in key if k["sid"] not in got]
    if missing:
        sys.exit(f"missing judgments: {missing}")
    H = {x["id"]: x for x in map(json.loads, open(human_path, encoding="utf-8"))}
    rows = []
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for k in key:
            g = got[k["sid"]]
            r = {"id": k["id"], "kind": k["kind"], "label": k["label"], "valid": bool(g["valid"]),
                 "reason": None if g["valid"] else g["reason"], "model": model, "interface": "claude.ai chat, incognito, default settings",
                 "batch": (int(k["sid"][1:]) - 1) // BATCH + 1, "date": date}
            rows.append(r); f.write(json.dumps(r, ensure_ascii=False) + "\n")
    S = {"n": len(rows), "batch": BATCH, "n-batches": len({r["batch"] for r in rows}), "model": model}
    for kind, tag in (("human", "h"), ("machine", "m")):
        k = [r for r in rows if r["kind"] == kind]
        S[f"n-{tag}"] = len(k)
        S[f"valid-{tag}"] = 100 * sum(r["valid"] for r in k) / len(k)
    both = [r for r in rows if r["id"] in H]
    po, kp = kappa([r["valid"] for r in both], [H[r["id"]]["valid"] for r in both])
    S.update({"n-both": len(both), "agree": 100 * po, "kappa": kp, "date": date})
    json.dump(S, open(os.path.join(os.path.dirname(out_jsonl) or ".", "llm_screen_summary.json"), "w"), indent=1)
    fmt = {"valid-h": "{:.1f}", "valid-m": "{:.1f}", "agree": "{:.1f}", "kappa": "{:.2f}"}
    month = datetime.date.fromisoformat(date).strftime("%-d %b %Y")
    with open(os.path.join(paper_dir, "results_screen.tex"), "w", encoding="utf-8") as f:
        f.write("\\makeatletter\n")
        for k, v in S.items():
            s = month if k == "date" else (fmt[k].format(v) if k in fmt else str(v))
            f.write(f"\\@namedef{{res@scr/{k}}}{{{s}}}\n")
        f.write("\\makeatother\n")
    lines = INSTRUCTIONS.format(n=BATCH).splitlines()
    with open(os.path.join(paper_dir, "screen_prompt.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{quote}\\small\\raggedright\n")
        f.write("\\\\\n".join(esc(l) for l in lines) + "\\\\\n")
        f.write("\\texttt{\\#\\#\\# I1 | HUMAN}, then the text; one such block per item.\n")
        f.write("\\end{quote}\n")
    print(json.dumps(S, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--items", default="audit/items100.json")
    m.add_argument("--out", default="audit/screen_batches")
    s = sub.add_parser("score")
    s.add_argument("--batches", default="audit/screen_batches")
    s.add_argument("--replies", default="audit/screen_batches/reply_*.txt")
    s.add_argument("--human", default="audit/audit1_human.jsonl")
    s.add_argument("--out", default="results/llm_screen.jsonl")
    s.add_argument("--paper", default="paper")
    s.add_argument("--date", default=datetime.date.today().isoformat())
    s.add_argument("--model", default="Claude Opus 5.5")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.items, a.out)
    else:
        score(a.batches, a.replies, a.human, a.out, a.paper, a.date, a.model)


if __name__ == "__main__":
    main()
