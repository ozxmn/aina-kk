import glob, json, os, sys
PREFIX = os.environ.get("PREFIX", "zs_")
out_dir, old_eval, old_dir, new_dirs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
cur = {r["id"]: r["text"] for r in map(json.loads, open("data/eval/eval.jsonl", encoding="utf-8"))}
old = {r["id"]: r["text"] for r in map(json.loads, open(old_eval, encoding="utf-8"))}
same = {i for i, t in cur.items() if old.get(i) == t}
os.makedirs(out_dir, exist_ok=True)
for fn in sorted(glob.glob(f"{old_dir}/{PREFIX}*.jsonl")):
    name = os.path.basename(fn); got = {}
    for l in open(fn):
        d = json.loads(l)
        if d["id"] in same: got[d["id"]] = l
    for nd in new_dirs:
        p = f"{nd}/{name}"
        if not os.path.exists(p): continue
        for l in open(p):
            d = json.loads(l)
            if d["id"] in cur: got[d["id"]] = l
    with open(f"{out_dir}/{name}", "w") as f:
        for i in cur:
            if i in got: f.write(got[i] if got[i].endswith("\n") else got[i] + "\n")
    print(name, "covered", len(got), "of", len(cur), "missing", len(cur) - len(got))
