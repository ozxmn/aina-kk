import glob, json, os, sys
from collections import Counter

def kappa(a, b):
    n = len(a)
    if n == 0: return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)

def rates(rows):
    out = {}
    for kind in ("human", "machine", "all"):
        S = [r for r in rows if kind == "all" or r["kind"] == kind]
        if S: out[kind] = {"n": len(S), "valid": sum(r["ok"] for r in S), "valid_rate": round(sum(r["ok"] for r in S) / len(S), 3),
                           "reasons": dict(Counter(r["reason"] for r in S if not r["ok"]))}
    return out

def main():
    if "--from-dir" in sys.argv:
        d = sys.argv[sys.argv.index("--from-dir") + 1]
        docs = []
        for fn in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            j = json.load(open(fn)); doc = j.get("data", j)
            docs.append({"id": j.get("id") or j.get("doc_id") or os.path.splitext(os.path.basename(fn))[0],
                         "kind": doc.get("kind"), "ok": doc.get("ok"), "reason": doc.get("reason"), "at": doc.get("at"),
                         "auditor": "human"})
        with open("results/human_audit.jsonl", "w", encoding="utf-8") as f:
            for r in docs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("wrote results/human_audit.jsonl", len(docs))
    AI = {r["id"]: r for r in map(json.loads, open("results/llm_audit.jsonl", encoding="utf-8"))}
    HU = {r["id"]: r for r in map(json.loads, open("results/human_audit.jsonl", encoding="utf-8"))} if os.path.exists("results/human_audit.jsonl") else {}
    res = {"ai_check_all": rates(list(AI.values()))}
    both = [i for i in HU if i in AI and HU[i]["ok"] is not None]
    if both:
        h = [int(HU[i]["ok"]) for i in both]; a = [int(AI[i]["ok"]) for i in both]
        res["human_audit"] = rates([HU[i] for i in both])
        res["ai_check_on_same_items"] = rates([AI[i] for i in both])
        res["agreement"] = {"n": len(both), "raw": round(sum(x == y for x, y in zip(h, a)) / len(both), 3),
                            "cohen_kappa": None if kappa(h, a) is None else round(kappa(h, a), 3),
                            "both_invalid_same_reason": sum(1 for i in both if not HU[i]["ok"] and not AI[i]["ok"] and HU[i]["reason"] == AI[i]["reason"]),
                            "human_N_ai_Y": [i for i in both if not HU[i]["ok"] and AI[i]["ok"]],
                            "human_Y_ai_N": [i for i in both if HU[i]["ok"] and not AI[i]["ok"]]}
    json.dump(res, open("results/audit_agreement.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))

if __name__ == "__main__": main()
