import os, sys, json, glob, time, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.57", "accelerate", "sentencepiece", "tiktoken"], check=False)
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel, PreTrainedModel
INP = os.path.dirname(glob.glob("/kaggle/input/**/eval.jsonl", recursive=True)[0]); OUT = "/kaggle/working"
rows = [json.loads(l) for l in open(f"{INP}/eval.jsonl", encoding="utf-8")]
DEV = "cuda:0"
ONLY = "__ONLY__"
MODELS = [("hc3-roberta", "Hello-SimpleAI/chatgpt-detector-roberta", 1, "seqcls"),
          ("fakespot-roberta", "fakespot-ai/roberta-base-ai-text-detection-v1", 1, "seqcls"),
          ("openai-roberta", "openai-community/roberta-base-openai-detector", 0, "seqcls"),
          ("tmr-roberta", "Oxidane/tmr-ai-text-detector", 1, "seqcls"),
          ("desklib-deberta", "desklib/ai-text-detector-v1.01", None, "desklib")]
class DesklibAIDetectionModel(nn.Module):
    def __init__(self, repo):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        cfg = AutoConfig.from_pretrained(repo)
        self.model = AutoModel.from_config(cfg)
        self.classifier = nn.Linear(cfg.hidden_size, 1)
        res = self.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")), strict=False)
        print("desklib load: missing", len(res.missing_keys), res.missing_keys[:5], "unexpected", len(res.unexpected_keys), res.unexpected_keys[:5], flush=True)
    def forward(self, input_ids, attention_mask=None, **kw):
        h = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).expand(h.size()).float()
        return {"logits": self.classifier((h * m).sum(1) / m.sum(1).clamp(min=1e-9))}
for tag, repo, mi, kind in MODELS:
    if ONLY and tag != ONLY: continue
    out = f"{OUT}/ots_{tag}.jsonl"
    if os.path.exists(out): continue
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(repo)
        model = DesklibAIDetectionModel(repo).to(DEV).eval() if kind == "desklib" else AutoModelForSequenceClassification.from_pretrained(repo).to(DEV).eval().half()
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
        with open(out, "w") as f, torch.no_grad():
            for b in range(0, len(order), 32):
                idx = order[b:b + 32]
                enc = tok([rows[i]["text"] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEV)
                lo = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])["logits"].float() if kind == "desklib" else model(**enc)["logits"].float()
                p = torch.sigmoid(lo[:, 0]) if kind == "desklib" else torch.softmax(lo, -1)[:, mi]
                for i, v in zip(idx, p.tolist()): f.write(json.dumps({"id": rows[i]["id"], "p_machine": v}) + "\n")
        del model; torch.cuda.empty_cache()
        print("done", tag, round(time.time() - t0), flush=True)
    except Exception as e:
        print("FAILED", tag, repr(e)[:300], flush=True)
print("ALL DONE", flush=True)
