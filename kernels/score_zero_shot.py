import os, sys, json, glob, time, math, random, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5.5.0", "accelerate"], check=False)
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
INP = os.path.dirname(glob.glob("/kaggle/input/**/eval.jsonl", recursive=True)[0]); OUT = "/kaggle/working"
sys.path.insert(0, INP)
from zs import token_stats, scores_single, binoculars
rows = [json.loads(l) for l in open(f"{INP}/eval.jsonl", encoding="utf-8")]
print("rows", len(rows), "gpus", torch.cuda.device_count(), flush=True)
PART = "__PART__"
SCORERS = [("llama3.2-3b", "unsloth/Llama-3.2-3B", "unsloth/Llama-3.2-3B-Instruct", 4, "A"),
           ("qwen3-4b", "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-4B", 2, "A"),
           ("mgpt-1.3b", "ai-forever/mGPT", "ai-forever/mGPT-1.3B-kazakh", 4, "B"),
           ("gemma4-e2b", "google/gemma-4-E2B", "google/gemma-4-E2B-it", 1, "B")]
def load(repo, dev):
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).to(dev).eval(); return m
def run_scorer(name, obs_repo, perf_repo, bs=4, max_len=512):
    out = f"{OUT}/zs_{name}.jsonl"
    if os.path.exists(out): return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(obs_repo); tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    d0, d1 = "cuda:0", ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    obs, perf = load(obs_repo, d0), load(perf_repo, d1)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    def score_batch(texts):
        lo, lab, mk = token_stats(obs, tok, texts, max_len, d0)
        so = scores_single(lo, lab, mk)
        lp, lab1, mk1 = token_stats(perf, tok, texts, max_len, d1)
        sp = scores_single(lp, lab1, mk1)
        bino = binoculars(lo, lp, lab1, mk1)
        del lo, lp
        return so, sp, bino
    n_oom = 0
    with open(out, "w") as f:
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]; texts = [rows[i]["text"] for i in idx]
            try:
                parts = [(idx, score_batch(texts))]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); n_oom += 1
                parts = [([i], score_batch([rows[i]["text"]])) for i in idx]
            for pidx, (so, sp, bino) in parts:
                for j, i in enumerate(pidx):
                    rec = {"id": rows[i]["id"], "n_tok": int(so["n_tok"][j])}
                    for k in ("ll", "logrank", "entropy", "lrr", "fdg"):
                        rec[f"obs_{k}"] = float(so[k][j]); rec[f"perf_{k}"] = float(sp[k][j])
                    rec["binoculars"] = float(bino[j]); f.write(json.dumps(rec) + "\n")
            if b % 400 == 0: print(name, b, len(order), round(time.time() - t0), "s", "oom_fallbacks", n_oom, flush=True)
    del obs, perf; torch.cuda.empty_cache(); print("done", name, round(time.time() - t0), flush=True)
def supervised(repo, tag, seed, epochs=2, lr=2e-5, bs=16, max_len=512, fp16=True):
    out = f"{OUT}/sup_{tag}_s{seed}.jsonl"
    if os.path.exists(out): return
    random.seed(seed); torch.manual_seed(seed)
    tr = [r for r in rows if r["split"] == "train" and r["variant"] == "orig" and r["label"] in ("H", "M") and not r["heldout"]]
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=2, dtype=torch.float32).to("cuda:0")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr) / bs); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=fp16); t0 = time.time(); n_nan = 0
    for ep in range(epochs):
        random.shuffle(tr); model.train()
        for b in range(0, len(tr), bs):
            batch = tr[b:b + bs]
            enc = tok([r["text"] for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda:0")
            y = torch.tensor([0 if r["label"] == "H" else 1 for r in batch], device="cuda:0")
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = F.cross_entropy(model(**enc).logits.float(), y)
            if not torch.isfinite(loss): n_nan += 1; opt.zero_grad(); sch.step(); continue
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        print(tag, seed, "epoch", ep, "loss", float(loss), "nan_steps", n_nan, round(time.time() - t0), "s", flush=True)
    model.eval()
    with open(out, "w") as f, torch.no_grad():
        for b in range(0, len(rows), 64):
            batch = rows[b:b + 64]
            enc = tok([r["text"] for r in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda:0")
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                p = torch.softmax(model(**enc).logits.float(), -1)[:, 1]
            for r, v in zip(batch, p.tolist()): f.write(json.dumps({"id": r["id"], "p_machine": v}) + "\n")
    del model; torch.cuda.empty_cache(); print("done", tag, seed, round(time.time() - t0), flush=True)
if PART == "B":
    for seed in (1, 2, 3):
        supervised("FacebookAI/xlm-roberta-base", "xlmr-base", seed)
        supervised("microsoft/mdeberta-v3-base", "mdeberta-v3", seed, fp16=True, bs=8)
for name, o, p, bs, part in SCORERS:
    if part == PART.rstrip("Z"):
        try: run_scorer(name, o, p, bs=bs)
        except Exception as e: print("SCORER FAILED", name, repr(e)[:300], flush=True)
print("ALL DONE", flush=True)
