import os, sys, glob, json, time, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.57", "accelerate", "sentencepiece", "tiktoken"], check=False)
import torch, transformers, numpy
INP = os.path.dirname(glob.glob("/kaggle/input/**/rows.jsonl", recursive=True)[0])
W = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/aina-kk-det/seed1/config.json", recursive=True)[0]))
sys.path.insert(0, INP)
import aina_kk_det as D
print("torch", torch.__version__, "transformers", transformers.__version__, "numpy", numpy.__version__, "gpus", torch.cuda.device_count(), flush=True)
rows = [json.loads(l) for l in open(f"{INP}/rows.jsonl", encoding="utf-8")]
prep = [D.prepare(r["raw"]) for r in rows]
texts = [p[0] for p in prep]
det = D.AinaKKDet(W)
t0 = time.time()
c = det.classifier_scores(texts)
print("classifier done", round(time.time() - t0), flush=True)
det.unload()
b = []
for k in range(0, len(texts), 500):
    b += det.binoculars_scores(texts[k:k + 500])
    print("binoculars", len(b), len(texts), round(time.time() - t0), flush=True)
with open("/kaggle/working/release_scores.jsonl", "w", encoding="utf-8") as f:
    for r, (t, s), ci, bi in zip(rows, prep, c, b):
        f.write(json.dumps({"id": r["id"], "classifier": ci, "binoculars": bi, "n_words": len(t.split()), "script": s}) + "\n")
json.dump({"torch": torch.__version__, "transformers": transformers.__version__, "numpy": numpy.__version__}, open("/kaggle/working/versions.json", "w"))
print("ALL DONE", round(time.time() - t0), flush=True)
