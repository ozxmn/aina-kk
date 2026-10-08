import glob, json, os, sys
PREFIX = os.environ.get("PREFIX", "zs_")
old_eval, score_dir, out = sys.argv[1:4]
old = {r["id"]: r["text"] for r in map(json.loads, open(old_eval, encoding="utf-8"))}
have = None
for fn in glob.glob(f"{score_dir}/{PREFIX}*.jsonl"):
    ids = {json.loads(l)["id"] for l in open(fn)}
    have = ids if have is None else have & ids
have = have or set()
n = 0
with open(out, "w", encoding="utf-8") as f:
    for l in open("data/eval/eval.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if old.get(r["id"]) != r["text"] or r["id"] not in have:
            f.write(l); n += 1
print("delta rows", n)
