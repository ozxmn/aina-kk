import os, sys, json, glob, time, math, random, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.57", "accelerate", "sentencepiece", "tiktoken"], check=False)
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
PART = "__PART__"
SMOKE = os.environ.get("AINA_SMOKE") == "1"
INP = os.path.dirname(glob.glob("/kaggle/input/**/eval.jsonl", recursive=True)[0]) if not SMOKE else os.environ["AINA_INP"]
OUT = "/kaggle/working" if not SMOKE else os.environ["AINA_OUT"]
sys.path.insert(0, INP)
from translit2021 import to_latin
from textnorm import CYR2LAT, to_sentence_end
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
rows = [json.loads(l) for l in open(f"{INP}/eval.jsonl", encoding="utf-8")]
if SMOKE: rows = rows[:3000]
orig = [r for r in rows if r["variant"] == "orig"]
GENS = sorted({r["generator"] for r in orig if r["label"] == "M"})
HELD = sorted({r["generator"] for r in orig if r["label"] == "M" and r["heldout"]})
train_clusters = sorted({r["cluster"] for r in orig if r["label"] == "H" and r["src_split"] == "train" and r["cluster"] is not None})
rng0 = random.Random(5); DEV_CL = set(rng0.sample(train_clusters, max(1, len(train_clusters) // 5)))
json.dump({"dev_clusters": sorted(DEV_CL), "gens": GENS, "held": HELD}, open(f"{OUT}/det_meta.json", "w"))
print("rows", len(rows), "gens", GENS, "dev clusters", len(DEV_CL), flush=True)

def train_pool(gens):
    return [r for r in orig if r["src_split"] == "train" and r["cluster"] not in DEV_CL and
            (r["label"] == "H" or (r["label"] == "M" and r["generator"] in gens))]
def homoglyphs(t, frac, rr):
    return "".join(CYR2LAT[c] if c in CYR2LAT and rr.random() < frac else c for c in t)
def augment(t, rr, aug):
    if not aug: return t
    x = rr.random()
    if x < 0.25: return to_latin(t)
    if x < 0.45: return homoglyphs(t, rr.choice([0.05, 0.1, 0.2]), rr)
    return t
def lencut(t, L):
    w = t.split()
    if L >= len(w): return t
    s = " ".join(w[:L]); c = to_sentence_end(s)
    return c if len(c.split()) >= 0.6 * L else s
def run(tag, gens, aug, seed, repo="microsoft/mdeberta-v3-base", epochs=2, lr=2e-5, bs=8, accum=2, max_len=512, lenaug=False, save=None):
    out = f"{OUT}/det_{tag}_s{seed}.jsonl"
    if os.path.exists(out): return
    t0 = time.time(); random.seed(seed); torch.manual_seed(seed); rr = random.Random(seed)
    tr = train_pool(gens)
    if SMOKE: tr = tr[:16]
    mlen = [len(r["text"].split()) for r in tr if r["label"] != "H"] or [150]
    def prep(r):
        t = r["text"]
        if lenaug:
            x = rr.random()
            if r["label"] == "H" and x < 0.5: t = lencut(t, rr.choice(mlen))
            elif x > 0.85: t = lencut(t, rr.randint(40, 120))
        return augment(t, rr, aug)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=2, dtype=torch.float32).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr) / (bs * accum)); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), max(1, steps))
    amp = DEV.startswith("cuda"); scaler = torch.amp.GradScaler(enabled=amp); n_nan = 0
    for ep in range(epochs):
        random.shuffle(tr); model.train(); opt.zero_grad()
        for k, b in enumerate(range(0, len(tr), bs)):
            batch = tr[b:b + bs]
            enc = tok([prep(r) for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEV)
            y = torch.tensor([0 if r["label"] == "H" else 1 for r in batch], device=DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(**enc).logits.float(), y) / accum
            if not torch.isfinite(loss): n_nan += 1; opt.zero_grad(); continue
            scaler.scale(loss).backward()
            if (k + 1) % accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
        print(tag, seed, "epoch", ep, "n_train", len(tr), "loss", float(loss) * accum, "nan", n_nan, round(time.time() - t0), "s", flush=True)
    model.eval()
    if save:
        os.makedirs(save, exist_ok=True); model.save_pretrained(save, safe_serialization=True); tok.save_pretrained(save)
    ev = rows if not SMOKE else rows[:32]
    order = sorted(range(len(ev)), key=lambda i: len(ev[i]["text"]))
    with open(out, "w") as f, torch.no_grad():
        for b in range(0, len(order), 64):
            idx = order[b:b + 64]
            enc = tok([ev[i]["text"] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                p = torch.softmax(model(**enc).logits.float(), -1)[:, 1]
            for i, v in zip(idx, p.tolist()): f.write(json.dumps({"id": ev[i]["id"], "p_machine": v}) + "\n")
    del model; torch.cuda.empty_cache() if amp else None
    print("done", tag, seed, round(time.time() - t0), flush=True)

INDIST = [g for g in GENS if g not in HELD]
if PART == "A":
    for seed in (1, 2, 3):
        run("base", INDIST, False, seed)
        run("aug", INDIST, True, seed)
        run("auglen", INDIST, True, seed, lenaug=True)
        run("xlmr", INDIST, False, seed, repo="FacebookAI/xlm-roberta-base")
elif PART == "B":
    for seed in (1, 2, 3):
        run("all", GENS, True, seed)
        run("allnoaug", GENS, False, seed)
        run("alllen", GENS, True, seed, lenaug=True)
    for g in INDIST:
        run("logo-" + g, [x for x in INDIST if x != g], True, 1, lenaug=True)
elif PART == "C":
    for seed in (1, 2, 3):
        run("auglen", INDIST, True, seed, lenaug=True)
        run("alllen", GENS, True, seed, lenaug=True)
elif PART == "K":
    KZ = [g for g in GENS if "KazLLM" in g or "Sherkala" in g]
    for seed in (1, 2, 3):
        run("kznoaug", INDIST + KZ, False, seed)
        run("kz", INDIST + KZ, True, seed)
        run("kzlen", INDIST + KZ, True, seed, lenaug=True)
elif PART == "W":
    for seed in (1, 2, 3):
        run("auglen", INDIST, True, seed, lenaug=True, save=f"{OUT}/aina-kk-det/seed{seed}")
elif PART == "SMOKE":
    run("smoke", INDIST, True, 1, lenaug=True)
print("ALL DONE", flush=True)
