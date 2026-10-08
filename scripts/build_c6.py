import json, random, sys
sys.path.insert(0, "scripts")
from prompts import C6, pick
gen = sys.argv[1]; n = int(sys.argv[2]); rng = random.Random(61)
M = [json.loads(l) for l in open("data/machine/machine_pool.jsonl", encoding="utf-8")]
cand = [m for m in M if m["label"] == "M" and m["generator"] != gen and m["condition"] in ("C1", "C2")]
rng.shuffle(cand)
by = {}
for m in cand: by.setdefault(m["generator"], []).append(m)
order = []
while len(order) < n and any(by.values()):
    for g in list(by):
        if by[g]: order.append(by[g].pop())
out = f"data/machine/{gen.replace(':','__').replace('/','_')}.c6.tasks.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for m in order[:n]:
        f.write(json.dumps({"task_id": f"{gen}|C6|{m['id']}", "generator": gen, "condition": "C6", "source_id": m["id"], "orig_generator": m["generator"],
                            "group": m["group"], "domain": m["domain"], "split": m["split"], "cluster": m.get("cluster"), "temperature": 0.9, "top_p": 0.95,
                            "gen_cfg": {}, "prompt": pick(C6, m["split"], rng).format(text=m["text"])}, ensure_ascii=False) + "\n")
print(out, min(n, len(order)))
