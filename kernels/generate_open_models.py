import os, json, time, sys, glob, datetime
INP = glob.glob("/kaggle/input/*aina-kk-gen-inputs*")[0]
OUT = "/kaggle/working"
try:
    from kaggle_secrets import UserSecretsClient
    HF = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("NO HF_TOKEN SECRET ATTACHED:", e); sys.exit(0)
os.environ["HF_TOKEN"] = HF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
print(torch.__version__, torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
SHERKALA_TMPL = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role']+'<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %} {% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run_llm(repo, tasks_file, out_file, bs=12, max_new=1300, rep=1.05):
    tasks = [json.loads(l) for l in open(f"{INP}/{tasks_file}", encoding="utf-8")]
    done = set()
    if os.path.exists(out_file):
        done = {json.loads(l)["task_id"] for l in open(out_file, encoding="utf-8")}
    tasks = [t for t in tasks if t["task_id"] not in done]
    if not tasks: return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo, token=HF)
    if "Sherkala" in repo: tok.chat_template = SHERKALA_TMPL
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="auto", token=HF)
    model.eval()
    eos = list({tok.convert_tokens_to_ids("<|eot_id|>"), tok.eos_token_id} - {None})
    print("loaded", repo, round(time.time() - t0), "s", flush=True)
    for t in tasks: t["_chat"] = tok.apply_chat_template([{"role": "user", "content": t["prompt"]}], tokenize=False, add_generation_prompt=True)
    by_temp = {}
    for t in tasks: by_temp.setdefault(t["temperature"], []).append(t)
    with open(out_file, "a", encoding="utf-8") as f:
        for temp, ts in by_temp.items():
            ts.sort(key=lambda t: len(t["_chat"]))
            for i in range(0, len(ts), bs):
                b = ts[i:i + bs]
                enc = tok([t["_chat"] for t in b], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.95,
                                         repetition_penalty=rep, eos_token_id=eos, pad_token_id=tok.pad_token_id)
                for t, g in zip(b, gen):
                    new = g[enc["input_ids"].shape[1]:]
                    text = tok.decode(new, skip_special_tokens=True).strip()
                    rec = {k: v for k, v in t.items() if k not in ("prompt", "_chat")}
                    rec.update({"status": 200, "text": text, "ts": now(), "meta": {"model": repo, "new_tokens": int((new != tok.pad_token_id).sum()),
                                "decoding": {"temperature": temp, "top_p": 0.95, "repetition_penalty": rep, "dtype": "float16"}}})
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush(); print(repo.split("/")[-1], "temp", temp, "batch", i // bs, "done", round(time.time() - t0), "s", flush=True)
    del model; torch.cuda.empty_cache()
def run_nllb(tasks_file, out_file, repo="facebook/nllb-200-distilled-600M"):
    tasks = [json.loads(l) for l in open(f"{INP}/{tasks_file}", encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(repo, src_lang="rus_Cyrl")
    model = AutoModelForSeq2SeqLM.from_pretrained(repo, torch_dtype=torch.float16).to("cuda:0").eval()
    tgt = tok.convert_tokens_to_ids("kaz_Cyrl")
    with open(out_file, "w", encoding="utf-8") as f:
        for t in tasks:
            paras = [p for p in t["ru_source"].split("\n\n") if p.strip()]
            outp = []
            for p in paras:
                enc = tok(p, return_tensors="pt", truncation=True, max_length=400).to("cuda:0")
                with torch.no_grad():
                    g = model.generate(**enc, forced_bos_token_id=tgt, max_new_tokens=512, num_beams=4)
                outp.append(tok.batch_decode(g, skip_special_tokens=True)[0])
            rec = {k: v for k, v in t.items()}
            rec.update({"status": 200, "text": "\n\n".join(outp), "ts": now(), "meta": {"model": repo, "decoding": {"num_beams": 4}}})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("nllb done", flush=True)
run_nllb("nllb.tasks.jsonl", f"{OUT}/nllb.out.jsonl")
run_llm("inception42/Llama-3.1-Sherkala-8B-Chat", "sherkala.tasks.jsonl", f"{OUT}/sherkala.out.jsonl")
run_llm("issai/LLama-3.1-KazLLM-1.0-8B", "kazllm.tasks.jsonl", f"{OUT}/kazllm.out.jsonl")
print("ALL DONE", flush=True)
