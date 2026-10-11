import json, math, random
from collections import Counter, defaultdict

E = {}
for l in open("data/eval/eval.jsonl", encoding="utf-8"):
    r = json.loads(l)
    E[r["id"]] = r
items = json.load(open("audit/postfilter/items.json", encoding="utf-8"))
J = {x["id"]: x for x in map(json.loads, open("audit/postfilter/judgments.jsonl", encoding="utf-8"))}
old = {x["id"] for x in json.load(open("audit/items.json", encoding="utf-8"))}

assert all(i["id"] in E and E[i["id"]]["variant"] == "orig" for i in items), "every item is an original passage of the frozen benchmark"
assert all(i["id"] in J and J[i["id"]]["ok"] is not None for i in items), "every item is judged"
strata = Counter(i["stratum"] for i in items)
overlap_with_first_audit = sum(i["id"] in old for i in items)

def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]

out = {"n": len(items), "strata": dict(strata), "overlap_with_first_audit": overlap_with_first_audit}
for kind in ("human", "machine"):
    its = [i for i in items if i["kind"] == kind]
    k = sum(J[i["id"]]["ok"] for i in its)
    out[kind] = {"n": len(its), "valid": k, "valid_share": k / len(its), "valid_ci": wilson(k, len(its)),
                 "reasons": dict(Counter(J[i["id"]]["reason"] for i in its if not J[i["id"]]["ok"]))}
    by = defaultdict(lambda: [0, 0])
    for i in its:
        by[i["stratum"]][0] += J[i["id"]]["ok"]
        by[i["stratum"]][1] += 1
    out[kind]["by_stratum"] = {s: {"valid": v[0], "n": v[1]} for s, v in sorted(by.items())}

size = Counter(r["domain"] for r in E.values() if r["variant"] == "orig" and r["label"] == "H")
hum = defaultdict(list)
for i in items:
    if i["kind"] == "human":
        hum[E[i["id"]]["domain"]].append(int(J[i["id"]]["ok"]))
W = sum(size[d] for d in hum)
wshare = lambda h: sum(size[d] * sum(v) / len(v) for d, v in h.items()) / W
rng = random.Random(0)
boot = sorted(wshare({d: [rng.choice(v) for _ in v] for d, v in hum.items()}) for _ in range(2000))
out["human"]["weighted_valid_share"] = wshare(hum)
out["human"]["weighted_valid_ci"] = [boot[49], boot[1949]]

out["human"]["looks_mt"] = out["human"]["reasons"].get("looks_mt", 0)

P = {}
for name in ("fakespot-roberta", "tmr-roberta", "desklib-deberta", "openai-roberta"):
    P[name] = {json.loads(l)["id"]: json.loads(l)["p_machine"] for l in open(f"kaggle/out_ots_merged_F/ots_{name}.jsonl")}
hv = [i["id"] for i in items if i["kind"] == "human" and J[i["id"]]["ok"]]
hi = [i["id"] for i in items if i["kind"] == "human" and not J[i["id"]]["ok"]]
out["ots_flags"] = {n: {"valid": sum(p[i] >= 0.5 for i in hv), "n_valid": len(hv), "invalid": sum(p[i] >= 0.5 for i in hi), "n_invalid": len(hi)} for n, p in P.items()}
json.dump(out, open("results/postfilter_audit.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "strata"}, indent=1))
