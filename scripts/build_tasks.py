import argparse, json, random, re, sys, os
sys.path.insert(0, "scripts")
from prompts import C1_ENC, C1_GEN, C1_ESSAY, C2, C3, C4, pick
ap = argparse.ArgumentParser()
for k in ["c1", "c2", "c3", "c4"]: ap.add_argument("--" + k, type=int, default=0)
ap.add_argument("--gen", required=True); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default=None)
ap.add_argument("--pool", default="data/human/human.jsonl"); ap.add_argument("--split", default=None); ap.add_argument("--domain", default=None)
ap.add_argument("--temps", default="0.7,1.0"); ap.add_argument("--gen_cfg", default="{}"); ap.add_argument("--shuffle", action="store_true")
a = ap.parse_args()
rng = random.Random(a.seed)
H = [json.loads(l) for fn in a.pool.split(",") for l in open(fn, encoding="utf-8")]
if a.split: H = [h for h in H if h["split"] == a.split]
if a.domain: H = [h for h in H if h["domain"] in a.domain.split(",")]
import glob as _g
_used = set()
for _fn in _g.glob("data/machine/*.tasks.jsonl") + _g.glob("kaggle/gen_inputs/*.tasks.jsonl"):
    for _l in open(_fn, encoding="utf-8"):
        _t = json.loads(_l)
        if _t["generator"] == a.gen: _used.add(_t["source_id"])
H = [h for h in H if h["id"] not in _used]
temps = [float(x) for x in a.temps.split(",")]
def first_sentence(t, maxw=14):
    s = re.split(r"(?<=[.!?])\s", t.strip())[0]
    w = s.split(); return " ".join(w[:maxw]).rstrip(",;:") + ("…" if len(w) > maxw else "")
def prefix(t, target=50):
    sents = re.split(r"(?<=[.!?])\s+", t.replace("\n\n", " ").strip()); out = []
    for s in sents:
        out.append(s)
        if len(" ".join(out).split()) >= target: break
    return " ".join(out)
tasks = []
def add(cond, src, prompt, extra=None):
    t = {"task_id": f"{a.gen}|{cond}|{src['id']}", "generator": a.gen, "condition": cond, "source_id": src["id"], "group": src["group"],
         "domain": src["domain"], "split": src["split"], "cluster": src.get("cluster"), "temperature": rng.choice(temps), "top_p": 0.95,
         "gen_cfg": json.loads(a.gen_cfg), "prompt": prompt}
    if extra: t.update(extra)
    tasks.append(t)
pool = H[:]; rng.shuffle(pool)
it = iter(pool)
for _ in range(a.c1):
    s = next(it); n = rng.choice([150, 200, 250])
    if s["domain"] == "encyclopedic": p = pick(C1_ENC, s["split"], rng).format(title=s["title"], n=n)
    elif s["domain"] == "essay": p = pick(C1_ESSAY, s["split"], rng).format(cue=first_sentence(s["text"]), n=n)
    else: p = pick(C1_GEN, s["split"], rng).format(cue=first_sentence(s["text"]), n=n)
    add("C1", s, p, {"target_words": n})
for _ in range(a.c2):
    s = next(it); n = rng.choice([150, 200, 250]); pf = prefix(s["text"])
    add("C2", s, pick(C2, s["split"], rng).format(prefix=pf, n=n), {"target_words": n, "human_prefix": pf})
for _ in range(a.c3):
    s = next(it); add("C3", s, pick(C3, s["split"], rng).format(text=s["text"]), {"human_source": s["text"]})
if a.c4:
    ru = [json.loads(l) for l in open("data/human/ru_pool.jsonl", encoding="utf-8")]
    rng.shuffle(ru)
    for s in ru[: a.c4]:
        add("C4", s, pick(C4, s["split"], rng).format(ru=s["text"]), {"ru_source": s["text"]})
if a.shuffle: random.Random(a.seed + 1000).shuffle(tasks)
out = a.out or f"data/machine/{a.gen.replace(':', '__').replace('/', '_')}.tasks.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for t in tasks: f.write(json.dumps(t, ensure_ascii=False) + "\n")
print(out, len(tasks))
