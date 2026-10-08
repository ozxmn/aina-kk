import glob, json, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, "scripts")
from textnorm import strip_title, to_sentence_end, mixed_script_token, homoglyph_count, kz_ratio

CUES = {
    "nb_hyphen_U+2011": lambda t: "‑" in t,
    "narrow_nbsp_U+202F": lambda t: " " in t,
    "soft_hyphen_U+00AD": lambda t: "­" in t,
    "ellipsis_char": lambda t: "…" in t,
    "title_line": lambda t: strip_title(t)[1],
    "markdown_header": lambda t: bool(re.search(r"(?m)^\s{0,3}#{1,6}\s", t)),
    "bold_markup": lambda t: "**" in t,
    "trailing_fragment": lambda t: to_sentence_end(t.strip()) != t.strip(),
    "mixed_script_token": mixed_script_token,
    "latin_homoglyph": lambda t: homoglyph_count(t) > 0,
}

def main():
    H = [json.loads(l)["text"] for l in open("data/human/human.jsonl", encoding="utf-8")]
    M = []
    for fn in sorted(glob.glob("data/machine/*.out.jsonl")) + sorted(glob.glob("kaggle/out*/*.out.jsonl")):
        for l in open(fn, encoding="utf-8"):
            r = json.loads(l)
            if r.get("status") == 200 and r.get("text") and r.get("condition") in ("C1", "C2"):
                M.append((r["generator"].split(":")[-1].split("/")[-1], r["text"]))
    res = {}
    for k, f in CUES.items():
        hg = sum(bool(f(t)) for t in H)
        by = defaultdict(lambda: [0, 0])
        for g, t in M:
            by[g][1] += 1; by[g][0] += bool(f(t))
        mg = sum(v[0] for v in by.values())
        res[k] = {"human": hg, "n_human": len(H), "human_rate": round(hg / len(H), 4),
                  "machine": mg, "n_machine": len(M), "machine_rate": round(mg / max(1, len(M)), 4),
                  "by_generator": {g: round(v[0] / v[1], 3) for g, v in sorted(by.items())}}
    json.dump(res, open("results/surface_cues.json", "w"), indent=1, ensure_ascii=False)
    for k, v in res.items():
        print(f"{k:24s} human {v['human_rate']:.3f}  machine {v['machine_rate']:.3f}  {v['by_generator']}")

if __name__ == "__main__": main()
