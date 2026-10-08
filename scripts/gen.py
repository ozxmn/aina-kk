import argparse, json, os, time, threading, datetime, requests, concurrent.futures as cf
def call(task):
    prov, model = task["generator"].split(":", 1)
    cfg = task.get("gen_cfg", {})
    if prov == "gemini":
        gc = {"temperature": task["temperature"], "topP": task.get("top_p", 0.95), "maxOutputTokens": 4096}
        if "thinkingConfig" in cfg: gc["thinkingConfig"] = cfg["thinkingConfig"]
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          json={"contents": [{"role": "user", "parts": [{"text": task["prompt"]}]}], "generationConfig": gc},
                          headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=300)
        j = r.json()
        if r.status_code != 200: return r.status_code, j, None, {}
        c = (j.get("candidates") or [{}])[0]
        txt = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
        return 200, j, txt, {"modelVersion": j.get("modelVersion"), "usage": j.get("usageMetadata"), "finish": c.get("finishReason")}
    url = {"groq": "https://api.groq.com/openai/v1/chat/completions", "github": "https://models.github.ai/inference/chat/completions"}[prov]
    key = {"groq": "GROQ_API_KEY", "github": "GITHUB_TOKEN"}[prov]
    body = {"model": model, "messages": [{"role": "user", "content": task["prompt"]}], "temperature": task["temperature"],
            "top_p": task.get("top_p", 0.95), "max_tokens": 1000 if prov == "groq" else 3000, **cfg}
    r = requests.post(url, json=body, headers={"Authorization": f"Bearer {os.environ[key]}"}, timeout=300)
    try: j = r.json()
    except Exception: j = {"raw": r.text[:300]}
    if r.status_code != 200:
        j["_retry_after"] = r.headers.get("retry-after"); return r.status_code, j, None, {}
    ch = j["choices"][0]
    return 200, j, ch["message"].get("content") or "", {"model": j.get("model"), "usage": j.get("usage"), "finish": ch.get("finish_reason")}
class Limiter:
    def __init__(s, rpm): s.gap, s.next, s.lock = 60.0 / rpm, 0.0, threading.Lock()
    def wait(s):
        with s.lock:
            now = time.time(); t = max(now, s.next); s.next = t + s.gap
        time.sleep(max(0, t - now))
def run(task, lim, stop):
    st, raw, text, meta = -1, {}, None, {}
    for attempt in range(7):
        if stop.is_set(): return None
        lim.wait()
        try: st, raw, text, meta = call(task)
        except Exception as e: st, raw = -1, {"error": str(e)[:300]}
        if st == 200 or st in (400, 401, 403, 404): break
        ra = raw.get("_retry_after") if isinstance(raw, dict) else None
        msg = json.dumps(raw)
        if st == 429 and ("PerDay" in msg or "per day" in msg.lower() or "RPD" in msg or "tokens per d" in msg): stop.set(); break
        time.sleep(float(ra) + 1 if ra and str(ra).replace('.', '', 1).isdigit() else min(180, 15 * 2 ** attempt))
    return {**{k: v for k, v in task.items() if k != "prompt"}, "status": st, "text": text, "meta": meta,
            "error": (json.dumps(raw)[:400] if st != 200 else None), "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tasks"); ap.add_argument("--rpm", type=float, default=10)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--out", default=None); a = ap.parse_args()
    out = a.out or a.tasks.replace(".tasks.jsonl", ".out.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            d = json.loads(l)
            if d.get("status") == 200 and d.get("text"): done.add(d["task_id"])
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")]
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    lim, stop, lock, ok, bad = Limiter(a.rpm), threading.Event(), threading.Lock(), 0, 0
    with open(out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run, t, lim, stop) for t in todo]
        for fu in cf.as_completed(futs):
            rec = fu.result()
            if rec is None: continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                if rec["status"] == 200 and rec["text"]: ok += 1
                else: bad += 1; print("fail", rec["status"], (rec["error"] or "")[:160], flush=True)
                if (ok + bad) % 25 == 0: print(f"ok {ok} bad {bad}", flush=True)
    print(f"finished ok {ok} bad {bad} stop={stop.is_set()}", flush=True)
if __name__ == "__main__": main()
