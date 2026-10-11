import glob, hashlib, json, os, re
from collections import Counter, defaultdict
import numpy as np

OUT = "results/results.json"
INPUTS = ["data/eval/eval.jsonl", "data/human/human_clean.jsonl", "data/human/essay_clean.jsonl", "results/analysis_det.json",
          "results/surface_cues.json", "results/linguistic.json", "results/audit_agreement.json", "results/human_audit.jsonl",
          "results/llm_audit.jsonl", "audit/items.json", "results/qc_machine_log.json", "results/freeze_v1_sha256.txt",
          "results/validity.json", "results/dev_replication.json", "results/postfilter_audit.json", "audit/postfilter/items.json",
          "audit/postfilter/judgments.jsonl"]
V = {}

def put(key, v, s, kind, src):
    assert key not in V, f"duplicate key {key}"
    V[key] = {"v": v, "s": s, "kind": kind, "src": src}

def pct(x, d=1): return "--" if x is None else f"{100 * x:.{d}f}"
def f3(x): return "--" if x is None else f"{x:.3f}"
def f2(x): return "--" if x is None else f"{x:.2f}"
def tex(s): return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
def sha(fn): return hashlib.sha256(open(fn, "rb").read()).hexdigest()
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

frozen = dict(l.split()[::-1] for l in open("results/freeze_v1_sha256.txt") if l.strip())
assert sha("data/eval/eval.jsonl") == frozen["data/eval/eval.jsonl"], "eval.jsonl differs from the frozen v1 benchmark"
E = [json.loads(l) for l in open("data/eval/eval.jsonl", encoding="utf-8")]
EI = {r["id"]: r for r in E}
O = [r for r in E if r["variant"] == "orig"]
HC = [json.loads(l) for fn in ("data/human/human_clean.jsonl", "data/human/essay_clean.jsonl") for l in open(fn, encoding="utf-8")]
A = json.load(open("results/analysis_det.json"))
C, N = A["configs"], A["n"]
CUES = json.load(open("results/surface_cues.json"))
LING = json.load(open("results/linguistic.json"))
AUD = json.load(open("results/audit_agreement.json"))
QCM = json.load(open("results/qc_machine_log.json"))
HA = [json.loads(l) for l in open("results/human_audit.jsonl", encoding="utf-8")]
ITM = {x["id"]: x for x in json.load(open("audit/items.json", encoding="utf-8"))}

GK = {"LLama-3.1-KazLLM-1.0-8B": "kazllm", "Llama-3.1-Sherkala-8B-Chat": "sherkala", "gemini-3.5-flash-lite": "g35fl", "gemini-3.1-flash-lite": "g31fl",
      "gemma-4-26b-a4b-it": "gemma26", "gemma-4-31b-it": "gemma31", "gpt-oss-120b": "gptoss", "qwen3.8-27b": "qwen", "nllb-200-distilled-600M": "nllb"}
GEN_NAME = {"gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "gemma-4-31b-it": "Gemma 4 31B",
            "gemma-4-26b-a4b-it": "Gemma 4 26B-A4B", "gpt-oss-120b": "gpt-oss-120b", "qwen3.8-27b": "Qwen3.8-27B", "nllb-200-distilled-600M": "NLLB-200 (600M)",
            "Llama-3.1-Sherkala-8B-Chat": "Sherkala-8B-Chat", "LLama-3.1-KazLLM-1.0-8B": "KazLLM-1.0-8B"}
ACCESS = {"gemini": "Gemini", "gemma": "Gemini", "gpt-oss": "Groq", "qwen": "Groq", "nllb": "open", "sherkala": "open", "kazllm": "open"}

nH = sum(r["label"] == "H" for r in O)
nNonH = sum(r["label"] != "H" for r in O)
gens = sorted({r["generator"] for r in O if r["label"] != "H"})
put("corpus/n-human", nH, f"{nH:,}", "data", "eval.jsonl: orig rows with label H")
put("corpus/n-nonhuman", nNonH, f"{nNonH:,}", "data", "eval.jsonl: orig rows with label M, MIX, HP, MT or MH")
put("corpus/n-gen", len(gens), str(len(gens)), "data", "eval.jsonl: distinct generators of non-human orig rows")
lab = Counter(r["label"] for r in O)
for l_, n_ in lab.items():
    put(f"corpus/n-label/{l_}", n_, f"{n_:,}", "data", f"eval.jsonl: orig rows with label {l_}")
doms = sorted({r["domain"] for r in O if r["label"] == "H"})
put("corpus/n-domains", len(doms), WORD[len(doms)], "name", "eval.jsonl: domains of human orig rows")
put("corpus/n-domains-n", len(doms), str(len(doms)), "data", "eval.jsonl: domains of human orig rows")
outlets = {h.get("source") for h in HC if h["domain"] == "news"}
hosts = {h.get("source") for h in HC if h["domain"] == "essay"}
put("corpus/n-outlets", len(outlets), str(len(outlets)), "data", "human_clean.jsonl: distinct sources of news passages")
put("corpus/n-edu-hosts", len(hosts), str(len(hosts)), "data", "essay_clean.jsonl: distinct hosts of educational passages")
held = sorted({r["generator"] for r in O if r.get("heldout")})
train_gens = sorted({r["generator"] for r in O if r["label"] == "M" and r["split"] == "train"})
put("corpus/n-train-gens", len(train_gens), WORD[len(train_gens)], "name", "eval.jsonl: generators with train-split M rows")
put("corpus/n-heldout-gens", len(held), WORD[len(held)], "name", "eval.jsonl: generators flagged heldout")
n_gemma_held = sum("gemma-4" in g for g in held)
put("corpus/n-heldout-gemma", n_gemma_held, WORD[n_gemma_held], "name", "eval.jsonl: held-out generators of the Gemma-4 family")
n_held_rows = sum(1 for r in O if r.get("heldout"))
put("corpus/n-heldout-rows", n_held_rows, f"{n_held_rows:,}", "data", "eval.jsonl: orig rows of held-out generators (all splits and classes)")
for l_, nm_ in (("M", "m"), ("MIX", "mix")):
    n_ = sum(1 for r in O if r.get("heldout") and r["label"] == l_)
    put(f"corpus/n-heldout-{nm_}", n_, f"{n_:,}", "data", f"eval.jsonl: orig rows of held-out generators with label {l_}")
put("corpus/heldout-names", ", ".join(GEN_NAME.get(g, g) for g in held), tex(", ".join(GEN_NAME.get(g, g) for g in held)), "data", "eval.jsonl: held-out generators")
mix = sum(v.get("mixed_script", 0) for v in QCM["drops"].values())
mix35 = sum(v.get("mixed_script", 0) for g, v in QCM["drops"].items() if "3.5-flash-lite" in g)
put("qc/mixed-script-drops", mix, str(mix), "data", "qc_machine_log.json: drops with reason mixed_script")
put("qc/mixed-script-drops-g35fl", mix35, str(mix35), "data", "qc_machine_log.json: mixed_script drops of Gemini 3.5 Flash-Lite")
VC = Counter((r["label"], r["variant"]) for r in E)
for (l_, var), n_ in VC.items():
    if var != "orig":
        put(f"corpus/variant/{l_}/{var}", n_, f"{n_:,}", "data", f"eval.jsonl: rows with label {l_} and variant {var}")
tab1 = defaultdict(Counter)
for r in O:
    if r["label"] != "H":
        tab1[r["generator"]][r["label"]] += 1
TABLE1 = []
for g in sorted(tab1, key=lambda g: -sum(tab1[g].values())):
    is_held = g in held
    gray = not is_held and g not in train_gens
    row = {"gen": g, "name": GEN_NAME.get(g, g), "mark": "held" if is_held else ("gray" if gray else ""),
           "access": next((v for k, v in ACCESS.items() if k in g.lower()), "")}
    for l_ in ("M", "MIX", "HP", "MT", "MH"):
        key = f"corpus/gen/{GK.get(g, g)}/{l_}"
        put(key, tab1[g][l_], str(tab1[g][l_]), "data", f"eval.jsonl: orig rows of {g} with label {l_}")
        row[l_] = key
    TABLE1.append(row)

def flat(prefix, obj, src):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat(f"{prefix}/{str(GK.get(k, k)).replace('_', '-').replace('@', '-at-')}", v, f"{src}/{k}")
    elif isinstance(obj, bool) or obj is None:
        return
    elif isinstance(obj, (int, float)):
        low = prefix.lower()
        is_auc = ("auroc" in low or "/auc" in low or low.endswith(("/latin2021", "/latin2021canon", "/trunc100", "/trunc50"))) and "tpr" not in low and "fpr" not in low
        is_thr = "/thr" in low
        s = f"{obj:.4f}" if is_thr else (f3(obj) if is_auc else (f"{obj:,}" if isinstance(obj, int) else pct(obj)))
        put(prefix, obj, s, "data", src)
    elif isinstance(obj, list) and len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
        fmt = f3 if "auroc" in prefix.lower() else pct
        put(prefix + "/lo", obj[0], fmt(obj[0]), "data", src + "[0]")
        put(prefix + "/hi", obj[1], fmt(obj[1]), "data", src + "[1]")

for k, v in N.items():
    put(f"det/n/{k.replace('_', '-')}", v, f"{v:,}", "data", f"analysis_det.json:n/{k}")
for cfg, e in C.items():
    pre = f"ots/{cfg[4:]}" if cfg.startswith("ots_") else f"det/{cfg.replace('det_', '').replace('_', '-')}"
    flat(pre, e, f"analysis_det.json:configs/{cfg}")
for g, e in A.get("logo", {}).items():
    flat(f"det/logo/{GK.get(g, g)}", e, f"analysis_det.json:logo/{g}")
for g, e in A.get("complement", {}).items():
    flat(f"det/complement/{g}", e, f"analysis_det.json:complement/{g}")
n_ots = sum(1 for k_ in C if k_.startswith("ots_"))
put("ots/n", n_ots, WORD[n_ots], "name", "analysis_det.json: number of public detectors (ots_* configs)")
n_combo = sum(1 for k_ in C if k_.startswith("combo_"))
put("det/n-combos", n_combo, WORD[n_combo], "name", "analysis_det.json: number of OR-combinations computed (combo_* configs)")
put("det/n/dev-cluster-share", 0.2, "a fifth", "const", "build_eval.py / aina_kk_det.py: rng.sample(train_clusters, len // 5)")

OTS = ["ots_fakespot-roberta", "ots_openai-roberta", "ots_hc3-roberta", "ots_tmr-roberta", "ots_desklib-deberta"]
ZS = ["zs:gemma4-e2b:binoculars", "zs:gemma4-e2b:obs_fdg", "zs:llama3.2-3b:binoculars", "zs:llama3.2-3b:obs_fdg", "zs:qwen3-4b:binoculars",
      "zs:qwen3-4b:obs_fdg", "zs:mgpt-1.3b:obs_fdg", "zs:mgpt-1.3b:perf_fdg"]
PREV = 0.05

def ppv(tpr, fpr, pi):
    d = tpr * pi + fpr * (1 - pi)
    return tpr * pi / d if d > 0 else None

for k, op in [("combo_auglen+bino", "dev"), ("det_base", "0.5"), ("det_base", "dev")] + [(o, "0.5") for o in OTS]:
    e = C[k]
    name = k.replace("det_", "").replace("ots_", "ots-").replace("_", "-")
    t_all = (e[f"tpr@{op}"] * N["M_indist"] + e[f"tpr_heldout@{op}"] * N["M_heldout_topicdisjoint"]) / (N["M_indist"] + N["M_heldout_topicdisjoint"])
    pv = ppv(t_all, e[f"fpr@{op}"], PREV)
    src = f"PPV at prevalence {PREV}: TPR = count-weighted mean of tpr@{op} ({N['M_indist']}) and tpr_heldout@{op} ({N['M_heldout_topicdisjoint']}); FPR = fpr@{op}; {k}"
    put(f"ppv/{name}/{op}/5", pv, pct(pv), "data", src)
    put(f"fdr/{name}/{op}/5", None if pv is None else 1 - pv, pct(None if pv is None else 1 - pv, 0), "data", src + " (1 - PPV)")
    put(f"falseflags/{name}/{op}/per1000", 1000 * e[f"fpr@{op}"], f"{round(1000 * e[f'fpr@{op}']):d}", "data", f"1000 x fpr@{op} of {k}")
ots_f = {k: C[k]["fpr@0.5"] for k in OTS}
put("ots/maxfpr", max(ots_f.values()), pct(max(ots_f.values())), "data", "max fpr@0.5 over the five public detectors")
mx = max(C[k]["fpr_by_domain@0.5"]["essay"] for k in OTS)
put("ots/max-essay-fpr", mx, pct(mx), "data", "max fpr_by_domain@0.5/essay over the five public detectors")
mx = max(C[k]["tpr@dev"] for k in OTS)
put("ots/max-tpr-dev", mx, pct(mx), "data", "max tpr@dev over the five public detectors (threshold recalibrated on dev humans)")
zs = {k: C[k] for k in ZS}
for nm, val, src in (("zs/best-tpr", max(e["tpr@dev"] for e in zs.values()), "max tpr@dev over the eight zero-shot rows of Table 2"),
                     ("zs/max-fpr-dev", max(e["fpr@dev"] for e in zs.values()), "max fpr@dev over the eight zero-shot rows"),
                     ("zs/min-fpr-dev", min(e["fpr@dev"] for e in zs.values()), "min fpr@dev over the eight zero-shot rows"),
                     ("zs/second-tpr", sorted((e["tpr@dev"] for e in zs.values()), reverse=True)[1], "second-highest tpr@dev over the eight zero-shot rows")):
    put(nm, val, pct(val), "data", src)
for k, e in C.items():
    for m, rg in (e.get("seed_range") or {}).items():
        if isinstance(rg, list):
            lo, hi = pct(rg[0]), pct(rg[1])
            put(f"seedspan/{k.replace('det_', '').replace('_', '-')}/{m.replace('@', '-at-').replace('_', '-')}", rg, lo if lo == hi else f"{lo}--{hi}",
                "data", f"analysis_det.json:configs/{k}/seed_range/{m}")

def nw(r): return len(r["text"].split())

Hn = [nw(r) for r in O if r["label"] == "H"]
Mn = [nw(r) for r in O if r["label"] == "M"]
put("len/h-under120", float(np.mean([x < 120 for x in Hn])), pct(np.mean([x < 120 for x in Hn])), "data", "eval.jsonl: share of human orig passages under 120 words")
put("len/m-under120", float(np.mean([x < 120 for x in Mn])), pct(np.mean([x < 120 for x in Mn])), "data", "eval.jsonl: share of M orig passages under 120 words")
put("len/h-median", float(np.median(Hn)), str(int(np.median(Hn))), "data", "eval.jsonl: median words, human orig passages")
put("len/m-median", float(np.median(Mn)), str(int(np.median(Mn))), "data", "eval.jsonl: median words, M orig passages")
for gk, gname in (("sherkala", "Sherkala"), ("kazllm", "KazLLM")):
    v = [nw(r) for r in O if r["label"] == "M" and gname in r["generator"]]
    put(f"len/{gk}-under120", float(np.mean([x < 120 for x in v])), pct(np.mean([x < 120 for x in v])), "data", f"eval.jsonl: share of {gname} M passages under 120 words")
    put(f"len/{gk}-median", float(np.median(v)), str(int(np.median(v))), "data", f"eval.jsonl: median words of {gname} M passages")
GLOBAL = [g for g in {r["generator"] for r in O if r["label"] == "M"} if "KazLLM" not in g and "Sherkala" not in g]
for k in ("det_base", "det_aug", "det_auglen", "det_all", "det_alllen", "det_xlmr", "det_kznoaug", "det_kz", "det_kzlen"):
    if k not in C:
        continue
    fg = C[k]["flag_by_gen@dev"]
    mn = min(fg[x] for x in GLOBAL)
    put(f"flag/{k[4:]}/min-global", mn, pct(mn), "data", f"min over the {len(GLOBAL)} global generators of flag_by_gen@dev, {k}")
    put(f"flag/{k[4:]}/argmin-global", min(GLOBAL, key=lambda x: fg[x]), tex(GEN_NAME.get(min(GLOBAL, key=lambda x: fg[x]))), "data", f"generator attaining flag/{k[4:]}/min-global")

def ng(t, n=6):
    w = re.findall(r"\w+", t.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

byg = defaultdict(list)
for r in O:
    if r["label"] == "H":
        byg[r["group"]].append(ng(r["text"]))
for gk, gname in (("sherkala", "Sherkala"), ("kazllm", "KazLLM")):
    ov = []
    for r in O:
        if r["label"] == "M" and gname in r["generator"]:
            m = ng(r["text"])
            ov.append(max((len(m & h) / max(1, len(m)) for h in byg.get(r["group"], [])), default=0.0))
    put(f"overlap/{gk}/mean", float(np.mean(ov)), pct(np.mean(ov)), "data", f"eval.jsonl: mean share of word 6-grams of {gname} outputs found in their source human passage")
    put(f"overlap/{gk}/p95", float(np.percentile(ov, 95)), pct(np.percentile(ov, 95)), "data", f"eval.jsonl: 95th percentile of the same share, {gname}")
import difflib
sims, wsame, wtot, n_aligned = [], 0, 0, 0
for r in E:
    if r["variant"] == "latin2021canon":
        o, c = EI[r["base_id"]]["text"], r["text"]
        sims.append(difflib.SequenceMatcher(None, o, c, autojunk=False).ratio())
        wo, wc = o.split(), c.split()
        if len(wo) == len(wc):
            wsame += sum(x == y for x, y in zip(wo, wc)); wtot += len(wo); n_aligned += 1
put("canon/charsim", float(np.mean(sims)), pct(np.mean(sims)), "data", "eval.jsonl: mean difflib ratio, original vs. canonicalized Latin-2021 text")
put("canon/word-exact", wsame / wtot, pct(wsame / wtot), "data", "eval.jsonl: share of identical words, over canonicalized passages with the original word count")
put("canon/n", len(sims), f"{len(sims):,}", "data", "eval.jsonl: latin2021canon rows")
put("canon/n-aligned", n_aligned, f"{n_aligned:,}", "data", "eval.jsonl: latin2021canon rows with the original word count")
_h, _runs = 0.0, []
for f_ in sorted(glob.glob("kaggle/out*/*.log")):
    ts = [float(x) for x in re.findall(r'"time":([0-9.]+)', open(f_).read())]
    if ts:
        _h += max(ts) / 3600
        _runs.append(f_)
put("compute/gpu-hours", _h, str(int(round(_h))), "data", f"sum of session lengths of {len(_runs)} Kaggle logs (kaggle/out*/*.log)")
ft = []
for f_ in glob.glob("kaggle/out_det_final_*/*.log"):
    ft += [int(x) for x in re.findall(r'"data":"done [a-z0-9]+ \d ([0-9]+)', open(f_).read())]
put("compute/ft-run-minutes", float(np.median(ft)) / 60, WORD[int(round(np.median(ft) / 60))], "name", f"median of {len(ft)} 'done' times in the final fine-tuning logs (training and scoring)")
LG = A.get("logo", {})
glob_out = [x["tpr@dev"] for g, x in LG.items() if "KazLLM" not in g and "Sherkala" not in g]
put("logo/min-global-out", min(glob_out), pct(min(glob_out)), "data", "analysis_det.json: min logo tpr@dev over the four training generators")

tw = {}
for fn in glob.glob("data/machine/*.tasks.jsonl") + glob.glob("kaggle/gen_inputs/*.tasks.jsonl"):
    for l in open(fn, encoding="utf-8"):
        t = json.loads(l)
        if t.get("target_words"):
            tw[t["task_id"]] = t["target_words"]
ratio = defaultdict(list)
for l in open("data/machine/machine_pool.jsonl", encoding="utf-8"):
    p = json.loads(l)
    r = EI.get(p["id"])
    if r and r["variant"] == "orig" and r["label"] == "M" and p["task_id"] in tw:
        ratio[r["generator"]].append(nw(r) / tw[p["task_id"]])
med = {g: float(np.median(v)) for g, v in ratio.items()}
for g, m in med.items():
    put(f"lenratio/{GK.get(g, g)}/median", m, pct(m, 0), "data", f"machine_pool.jsonl + task files: median produced/requested words, {g} (n={len(ratio[g])})")
put("lenratio/min", min(med.values()), pct(min(med.values()), 0), "data", "min over generators of lenratio median")
put("lenratio/max", max(med.values()), pct(max(med.values()), 0), "data", "max over generators of lenratio median")
put("lenratio/argmin", min(med, key=med.get), tex(GEN_NAME.get(min(med, key=med.get))), "data", "generator with the lowest lenratio median")
nzs = len({k.split(':')[1] for k in C if k.startswith('zs:')})
put("corpus/n-zs-families", nzs, WORD[nzs], "name", "analysis_det.json: distinct zero-shot scoring families")

for c, v in CUES.items():
    put(f"cue/{c}/h", v["human_rate"], pct(v["human_rate"]), "data", f"surface_cues.json:{c}/human_rate")
    put(f"cue/{c}/m", v["machine_rate"], pct(v["machine_rate"]), "data", f"surface_cues.json:{c}/machine_rate")
    for g, x in v["by_generator"].items():
        put(f"cue/{c}/{GK.get(g, g)}", x, pct(x), "data", f"surface_cues.json:{c}/by_generator/{g}")
hh = next(iter(CUES.values()))
put("cue/n-human", hh["n_human"], f"{hh['n_human']:,}", "data", "surface_cues.json:n_human")
put("cue/n-machine", hh["n_machine"], f"{hh['n_machine']:,}", "data", "surface_cues.json:n_machine")
for i, o in enumerate(LING["features"]):
    put(f"ling/{i}/name", o["feature"], tex(o["feature"]), "data", f"linguistic.json:features[{i}]")
    put(f"ling/{i}/h", o["median_human"], f"{o['median_human']:.2f}", "data", f"linguistic.json:features[{i}]/median_human")
    put(f"ling/{i}/m", o["median_machine"], f"{o['median_machine']:.2f}", "data", f"linguistic.json:features[{i}]/median_machine")
    put(f"ling/{i}/d", o["cliffs_delta_machine_vs_human"], f"{o['cliffs_delta_machine_vs_human']:+.2f}", "data", f"linguistic.json:features[{i}]/cliffs_delta")
    put(f"ling/{i}/q", o["q_bh"], f"{o['q_bh']:.0e}", "data", f"linguistic.json:features[{i}]/q_bh")
put("ling/n-human", LING["n_human"], f"{LING['n_human']:,}", "data", "linguistic.json:n_human")
put("ling/n-machine", LING["n_machine"], f"{LING['n_machine']:,}", "data", "linguistic.json:n_machine")
hu, au, ag = AUD["human_audit"], AUD["ai_check_all"], AUD["agreement"]
put("aud/n-human-judged", hu["all"]["n"], str(hu["all"]["n"]), "data", "audit_agreement.json:human_audit/all/n")
put("aud/human-valid-h", hu["human"]["valid_rate"], pct(hu["human"]["valid_rate"]), "data", "audit_agreement.json:human_audit/human/valid_rate")
put("aud/human-valid-m", hu["machine"]["valid_rate"], pct(hu["machine"]["valid_rate"]), "data", "audit_agreement.json:human_audit/machine/valid_rate")
put("aud/n-llm", au["all"]["n"], str(au["all"]["n"]), "data", "audit_agreement.json:ai_check_all/all/n")
put("aud/llm-valid-h", au["human"]["valid_rate"], pct(au["human"]["valid_rate"]), "data", "audit_agreement.json:ai_check_all/human/valid_rate")
put("aud/llm-valid-m", au["machine"]["valid_rate"], pct(au["machine"]["valid_rate"]), "data", "audit_agreement.json:ai_check_all/machine/valid_rate")
put("aud/n-both", ag["n"], str(ag["n"]), "data", "audit_agreement.json:agreement/n")
put("aud/raw", ag["raw"], pct(ag["raw"]), "data", "audit_agreement.json:agreement/raw")
put("aud/kappa", ag["cohen_kappa"], f2(ag["cohen_kappa"]), "data", "audit_agreement.json:agreement/cohen_kappa")
ndis = len(ag["human_N_ai_Y"]) + len(ag["human_Y_ai_N"])
mt_m = [r for r in HA if r["kind"] == "machine" and r["reason"] == "looks_mt"]
mt_dis = sum(1 for r in mt_m if r["id"] in set(ag["human_N_ai_Y"]))
put("aud/n-disagree", ndis, str(ndis), "data", "audit_agreement.json: items judged differently by the annotator and the LLM screen")
put("aud/n-disagree-mt", mt_dis, str(mt_dis), "data", "human_audit.jsonl: machine items judged looks_mt by the annotator and valid by the LLM screen")
gm = lambda r: ITM[r["id"]]["generator"].split(":")[-1].split("/")[-1]
byg_mt = Counter(gm(r) for r in mt_m)
tot_m = Counter(gm(r) for r in HA if r["kind"] == "machine")
AUD_MT = []
for g in sorted(byg_mt, key=lambda g: (-byg_mt[g], g)):
    put(f"aud/mt/{GK.get(g, g)}/n", byg_mt[g], str(byg_mt[g]), "data", f"human_audit.jsonl: {g} items judged looks_mt")
    put(f"aud/mt/{GK.get(g, g)}/of", tot_m[g], str(tot_m[g]), "data", f"human_audit.jsonl: {g} machine items audited")
    AUD_MT.append(GK.get(g, g))

VAL = json.load(open("results/validity.json"))
sp = VAL["split"]
put("val/split/n-neardup", sp["test_humans_with_train_near_duplicate"], str(sp["test_humans_with_train_near_duplicate"]), "data", "validity.json:split/test_humans_with_train_near_duplicate")
for k_, nm_ in (("train_clusters", "train-clusters"), ("test_clusters", "test-clusters"), ("shared_clusters", "shared-clusters")):
    put(f"val/split/{nm_}", sp[k_], str(sp[k_]), "data", f"validity.json:split/{k_}")
for k_, nm_ in (("essay_train_clusters", "essay-train-clusters"), ("essay_test_clusters", "essay-test-clusters")):
    put(f"val/split/{nm_}", sp[k_], WORD[sp[k_]], "name", f"validity.json:split/{k_}")
for d_, n_ in sp["dev_humans_by_domain"].items():
    put(f"val/split/dev-{d_}", n_, str(n_), "data", f"validity.json:split/dev_humans_by_domain/{d_}")
th_ = VAL["threshold_theory"]
for k_ in ("mean", "lo", "hi"):
    put(f"val/thr/{k_}", th_[k_], pct(th_[k_]), "data", f"validity.json:threshold_theory/{k_} (true FPR of the empirical 99th percentile of {th_['n']} scores, 20,000 draws)")
put("val/thr/p-above", th_["p_above_1.5pct"], pct(th_["p_above_1.5pct"], 0), "data", "validity.json:threshold_theory/p_above_1.5pct")
su = VAL["surface"]
put("val/surface/n-features", su["n_features"], str(su["n_features"]), "data", "validity.json:surface/n_features")
for k_, nm_ in (("auroc", "auroc"), ("auroc_heldout", "auroc-heldout"), ("auroc_len120_220", "auroc-len120-220")):
    put(f"val/surface/{nm_}", su[k_], f3(su[k_]), "data", f"validity.json:surface/{k_}")
for k_, nm_ in (("fpr_at_dev", "fpr-at-dev"), ("tpr_at_dev", "tpr-at-dev"), ("tpr_heldout_at_dev", "tpr-heldout-at-dev")):
    put(f"val/surface/{nm_}", su[k_], pct(su[k_]), "data", f"validity.json:surface/{k_}")
dg = VAL["digits"]
put("val/digits/h", dg["human_share"], pct(dg["human_share"], 0), "data", "validity.json:digits/human_share")
put("val/digits/m", dg["machine_share"], pct(dg["machine_share"], 0), "data", "validity.json:digits/machine_share")
for k_, nm_ in (("n_human_no_digit", "n-h-nodigit"), ("n_human_no_digit_essay", "n-h-nodigit-essay")):
    put(f"val/digits/{nm_}", dg[k_], f"{dg[k_]:,}", "data", f"validity.json:digits/{k_}")
for t_, nm_ in (("det_base", "base"), ("combo", "combo")):
    put(f"val/digits/{nm_}/auroc-nodigit", dg[t_]["auroc_no_digit"], f3(dg[t_]["auroc_no_digit"]), "data", f"validity.json:digits/{t_}/auroc_no_digit")
    put(f"val/digits/{nm_}/flagged-nodigit", dg[t_]["flagged_no_digit"], str(dg[t_]["flagged_no_digit"]), "data", f"validity.json:digits/{t_}/flagged_no_digit")
    for k_, nm2_ in (("fpr_no_digit", "fpr-nodigit"), ("fpr_digit", "fpr-digit")):
        put(f"val/digits/{nm_}/{nm2_}", dg[t_][k_], pct(dg[t_][k_]), "data", f"validity.json:digits/{t_}/{k_}")
fa = VAL["family"]
for t_, nm_ in (("det_auglen", "auglen"), ("combo", "combo")):
    for k_, nm2_ in (("kazakh_built", "kz"), ("gemma4", "gemma")):
        put(f"val/family/{nm_}/{nm2_}", fa[t_][k_], pct(fa[t_][k_]), "data", f"validity.json:family/{t_}/{k_}")
put("val/family/n-kz", fa["n_kazakh_built"], str(fa["n_kazakh_built"]), "data", "validity.json:family/n_kazakh_built")
put("val/family/n-gemma", fa["n_gemma4"], str(fa["n_gemma4"]), "data", "validity.json:family/n_gemma4")
la = VAL["latin"]
put("val/latin/n", la["n_detectors"], str(la["n_detectors"]), "data", "validity.json:latin/n_detectors")
put("val/latin/n-low", la["n_detectors"] - la["n_fpr_above_10"], str(la["n_detectors"] - la["n_fpr_above_10"]), "data", "validity.json:latin n_detectors - n_fpr_above_10")
put("val/latin/max-tpr-low", la["max_tpr_when_fpr_below_10"], pct(la["max_tpr_when_fpr_below_10"]), "data", "validity.json:latin/max_tpr_when_fpr_below_10")
put("val/latin/n-high", la["n_fpr_above_10"], WORD[la["n_fpr_above_10"]], "name", "validity.json:latin/n_fpr_above_10")
put("val/latin/min-fpr-high", la["min_fpr_of_those"], pct(la["min_fpr_of_those"], 0), "data", "validity.json:latin/min_fpr_of_those")
put("val/latin/max-fpr-high", max(v_["fpr"] for v_ in la["by_detector"].values()), pct(max(v_["fpr"] for v_ in la["by_detector"].values()), 0), "data", "validity.json:latin max fpr over detectors")
tp_ = VAL["templates"]
put("val/templates/n", tp_["prompts_own_pool_only"], f"{tp_['prompts_own_pool_only']:,}", "data", "validity.json:templates/prompts_own_pool_only")
put("val/templates/other", tp_["prompts_other"], str(tp_["prompts_other"]), "data", "validity.json:templates/prompts_other")
for k_ in ("main_heldout_rows", "main_test_rows", "kz_gemma_rows", "dev_rows_any", "kz_rows"):
    put(f"val/pools/{k_.replace('_', '-')}", VAL["pools"][k_], str(VAL["pools"][k_]), "data", f"validity.json:pools/{k_}")
import datetime as _dt
for d_, e_ in VAL["dates"].items():
    put(f"val/dates/{d_}/dated", e_["dated"], f"{e_['dated']:,}", "data", f"validity.json:dates/{d_}/dated")
    put(f"val/dates/{d_}/n", e_["n"], f"{e_['n']:,}", "data", f"validity.json:dates/{d_}/n")
    if e_["max"]:
        dt_ = _dt.datetime.strptime(e_["max"][:10], "%Y-%m-%d")
        put(f"val/dates/{d_}/max", e_["max"], f"{dt_.day} {dt_.strftime('%b %Y')}", "data", f"validity.json:dates/{d_}/max")
DR = json.load(open("results/dev_replication.json"))
put("devrep/n-fit-clusters", DR["clusters"] - 1, WORD[DR["clusters"] - 1], "name", "dev_replication.json: clusters - 1 (thresholds fitted without the judged cluster)")
put("devrep/n-h", DR["n_dev_humans"], f"{DR['n_dev_humans']:,}", "data", "dev_replication.json:n_dev_humans")
put("devrep/n-m", DR["n_dev_machine"], f"{DR['n_dev_machine']:,}", "data", "dev_replication.json:n_dev_machine")
for row_, e_ in list(DR["rows"].items()) + [("combo-" + k_[4:], v_) for k_, v_ in DR["combos"].items()]:
    for m_, x_ in e_.items():
        if m_ == "n":
            continue
        put(f"devrep/{row_}/{m_.replace('_', '-')}", x_, pct(x_), "data", f"dev_replication.json:{row_}/{m_}")
for m_, n_ in DR["rows"]["combo"]["n"].items():
    put(f"devrep/n/{m_.replace('_', '-')}", n_, str(n_), "data", f"dev_replication.json:rows/combo/n/{m_}")

PF = json.load(open("results/postfilter_audit.json"))
put("aud/post/n", PF["n"], str(PF["n"]), "data", "postfilter_audit.json:n")
put("aud/post/n-h", PF["human"]["n"], str(PF["human"]["n"]), "data", "postfilter_audit.json:human/n")
put("aud/post/n-m", PF["machine"]["n"], str(PF["machine"]["n"]), "data", "postfilter_audit.json:machine/n")
per_h = {v["n"] for v in PF["human"]["by_stratum"].values()}; per_m = {v["n"] for v in PF["machine"]["by_stratum"].values()}
assert len(per_h) == 1 and len(per_m) == 1
put("aud/post/n-h-per-domain", per_h.pop(), str(PF["human"]["by_stratum"]["news"]["n"]), "data", "postfilter_audit.json:human/by_stratum/*/n")
put("aud/post/n-m-per-gen", per_m.pop(), str(next(iter(PF["machine"]["by_stratum"].values()))["n"]), "data", "postfilter_audit.json:machine/by_stratum/*/n")
put("aud/post/overlap-first", PF["overlap_with_first_audit"], str(PF["overlap_with_first_audit"]), "data", "postfilter_audit.json:overlap_with_first_audit")
put("aud/post/h-valid", PF["human"]["valid"], str(PF["human"]["valid"]), "data", "postfilter_audit.json:human/valid")
put("aud/post/h-valid-w", PF["human"]["weighted_valid_share"], pct(PF["human"]["weighted_valid_share"], 0), "data", "postfilter_audit.json:human/weighted_valid_share")
put("aud/post/h-valid-w-ci/lo", PF["human"]["weighted_valid_ci"][0], pct(PF["human"]["weighted_valid_ci"][0], 0), "data", "postfilter_audit.json:human/weighted_valid_ci[0]")
put("aud/post/h-valid-w-ci/hi", PF["human"]["weighted_valid_ci"][1], pct(PF["human"]["weighted_valid_ci"][1], 0), "data", "postfilter_audit.json:human/weighted_valid_ci[1]")
rh = PF["human"]["reasons"]
put("aud/post/h-broken", rh.get("broken", 0), str(rh.get("broken", 0)), "data", "postfilter_audit.json:human/reasons/broken")
put("aud/post/h-mt", rh.get("looks_mt", 0), str(rh.get("looks_mt", 0)), "data", "postfilter_audit.json:human/reasons/looks_mt")
oth = sum(v for k, v in rh.items() if k not in ("broken", "looks_mt"))
put("aud/post/h-other", oth, str(oth), "data", "postfilter_audit.json:human/reasons other than broken and looks_mt")
rm = PF["machine"]["reasons"]
put("aud/post/m-broken", rm.get("broken", 0), WORD.get(rm.get("broken", 0), str(rm.get("broken", 0))), "name", "postfilter_audit.json:machine/reasons/broken")
put("aud/post/m-mt", rm.get("looks_mt", 0), str(rm.get("looks_mt", 0)), "data", "postfilter_audit.json:machine/reasons/looks_mt")
for n_ in ("fakespot-roberta", "tmr-roberta"):
    f_ = PF["ots_flags"][n_]
    put(f"aud/post/{n_}/valid", f_["valid"], str(f_["valid"]), "data", f"postfilter_audit.json:ots_flags/{n_}/valid")
    put(f"aud/post/{n_}/invalid", f_["invalid"], str(f_["invalid"]), "data", f"postfilter_audit.json:ots_flags/{n_}/invalid")
f_ = PF["ots_flags"]["fakespot-roberta"]
put("aud/post/n-valid", f_["n_valid"], str(f_["n_valid"]), "data", "postfilter_audit.json:ots_flags/*/n_valid")
put("aud/post/n-invalid", f_["n_invalid"], str(f_["n_invalid"]), "data", "postfilter_audit.json:ots_flags/*/n_invalid")

CONST = [
    ("const/cutoff-date", "30 Nov 2022", "30 Nov 2022", "ext: ChatGPT release date; FineWeb2 crawl filter in fw2_news_local.py / fw2_essay_local.py"),
    ("const/span-min", 120, "120", "fineweb_news.py, cc100_sample.py: 120 <= n <= 320"),
    ("const/span-max", 320, "320", "fineweb_news.py, cc100_sample.py: 120 <= n <= 320"),
    ("const/glotlid-p", 0.9, "0.9", "qc_human.py, qc_essay.py, qc_machine.py: p < 0.9 dropped"),
    ("const/jaccard", 0.8, "0.8", "near-duplicate MinHash threshold"),
    ("const/xsplit-jaccard", 0.3, "0.3", "validity.py: MinHashLSH(threshold=0.3) between test and training humans"),
    ("const/id-digits", 12, "12", "textnorm.py: \\d{12} scrubbed as an ID number"),
    ("const/c2-prefix", 50, "50", "build_tasks.py: prefix(t, target=50)"),
    ("const/req-words", [150, 200, 250], "150, 200 or 250", "build_tasks.py: n = rng.choice([150, 200, 250])"),
    ("const/min-words", 60, "60", "qc_machine.py: drop if nw < 60"),
    ("const/max-words", 320, "320", "qc_machine.py: trim to <= 320 words"),
    ("const/hg-attack", 0.10, "10", "build_eval.py: homoglyph_attack(..., 0.10, ...)"),
    ("const/trunc-long", 100, "100", "build_eval.py: w[:100]"),
    ("const/trunc-short", 50, "50", "build_eval.py: w[:50]"),
    ("const/trunc-short-word", 50, "Fifty", "build_eval.py: w[:50]"),
    ("const/max-tokens", 512, "512", "aina_kk_det.py, aina_kk_eval.py: max_len=512"),
    ("const/epochs", 2, "2", "aina_kk_det.py: epochs=2"),
    ("const/lr", 2e-5, "$2\\times10^{-5}$", "aina_kk_det.py: lr=2e-5"),
    ("const/seeds", 3, "three", "aina_kk_det.py: for seed in (1, 2, 3)"),
    ("const/seeds-digit", 3, "3", "aina_kk_det.py: for seed in (1, 2, 3)"),
    ("const/default-thr", 0.5, "0.5", "probability threshold 0.5"),
    ("const/target-fpr", 0.01, "1", "analyze_det.py: thr(t, neg, fpr=0.01)"),
    ("const/ci-level", 0.95, "95", "analyze_det.py: percentiles 2.5 and 97.5"),
    ("const/boot-rate-B", 400, "400", "analyze_det.py: boot_rate(B=400, seed=0)"),
    ("const/boot-auc-B", 300, "300", "analyze_det.py: boot_auc(B=300, seed=0)"),
    ("const/lenctrl-lo", 120, "120", "analyze_det.py: 120 <= n_words < 220"),
    ("const/lenctrl-hi", 220, "220", "analyze_det.py: 120 <= n_words < 220"),
    ("const/lenmatch-h", 0.5, "half", "aina_kk_det.py: H and x < 0.5 -> machine length"),
    ("const/lenmatch-ex", 0.15, "15", "aina_kk_det.py: x > 0.85 -> 40-120-word excerpt"),
    ("const/excerpt-lo", 40, "40", "aina_kk_det.py: rr.randint(40, 120)"),
    ("const/excerpt-hi", 120, "120", "aina_kk_det.py: rr.randint(40, 120)"),
    ("const/aug-latin", 0.25, "0.25", "aina_kk_det.py: x < 0.25 -> Latin-2021"),
    ("const/aug-latin-words", 0.25, "a quarter", "aina_kk_det.py: x < 0.25 -> Latin-2021"),
    ("const/aug-hg", 0.20, "0.20", "aina_kk_det.py: 0.25 <= x < 0.45 -> homoglyphs"),
    ("const/aug-hg-words", 0.20, "a fifth", "aina_kk_det.py: 0.25 <= x < 0.45 -> homoglyphs"),
    ("const/aug-hg-rates", [5, 10, 20], "5, 10 or 20", "aina_kk_det.py: rr.choice([0.05, 0.1, 0.2])"),
    ("const/wd", 0.01, "0.01", "aina_kk_det.py: weight_decay=0.01"),
    ("const/warmup", 0.06, "6", "aina_kk_det.py: int(0.06 * steps)"),
    ("const/batch", 16, "16", "aina_kk_det.py: bs=8, accum=2"),
    ("const/microbatch", 8, "8", "aina_kk_det.py: bs=8"),
    ("const/accum", 2, "2", "aina_kk_det.py: accum=2"),
    ("const/clip", 1.0, "1.0", "aina_kk_det.py: clip_grad_norm_(..., 1.0)"),
    ("const/per-n", 1000, "1,000", "presentation: false flags per 1,000 human passages"),
    ("const/prevalence", PREV, "5", "build_results.py: PREV = 0.05"),
    ("const/ngram", 6, "6", "build_results.py: ng(t, n=6)"),
    ("const/gpu-gb", 16, "16", "Kaggle T4 / P100 GPUs (16 GB)"),
]
for key, v, s, chk in CONST:
    put(key, v, s, "const", chk)

EXT = [
    ("ext/turnitin-n-langs", 7, "seven", "Turnitin guide 28477544839821 (page updated 7 Oct 2026): long-form English, Spanish, Japanese, Modern Arabic, Dutch, German, Turkish", "VERIFIED 11 Oct 2026"),
    ("ext/turnitin-langs", "English, Spanish, Japanese, Arabic, Dutch, German and Turkish", "English, Spanish, Japanese, Arabic, Dutch, German and Turkish", "Turnitin guide 28477544839821; 'No report will be generated if the submitted content is not in one of the supported languages'", "VERIFIED 11 Oct 2026"),
    ("ext/turnitin-accessed", "11 Oct 2026", "11 Oct 2026", "access date of the Turnitin guide and the GPTZero, Copyleaks, Originality.ai and Pangram language pages (citations/verification.md)", "VERIFIED 11 Oct 2026"),
    ("ext/antiplagiat-claim", 0.01, "1", "antiplagiat.com news item, 7 Apr 2025: 'Число ложных срабатываний сведено к 1%'; works with Russian, English and Kazakh", "VERIFIED 11 Oct 2026"),
    ("ext/antiplagiat-date", "7 Apr 2025", "7 Apr 2025", "antiplagiat.com news item date '7 апреля 2025'", "VERIFIED 11 Oct 2026"),
    ("ext/statcounter-share", 0.94, "94", "StatCounter AI-chatbot market share, Kazakhstan, September 2026: ChatGPT 68.06% + Gemini 25.98% = 94.04%", "VERIFIED 11 Oct 2026"),
    ("ext/statcounter-month", "September 2026", "September 2026", "StatCounter period shown on the page", "VERIFIED 11 Oct 2026"),
    ("ext/coling25-kk-samples", 2471, "2,471", "wang2025genai (arXiv 2501.11012v2): per-language test sizes, KK 2,471; Kazakh in bold = unseen in training", "VERIFIED 11 Oct 2026 (PDF)"),
    ("ext/human-study-acc", 79.67, "79.67", "wang2026is (arXiv 2502.11614), App. D.7: '300 (human, GPT-4o) pairs ... detection accuracy of 79.67%'", "VERIFIED 11 Oct 2026 (PDF)"),
    ("ext/human-study-pairs", 300, "300", "wang2026is App. D.7 and Table 1 (Kazakh Wikipedia, 300)", "VERIFIED 11 Oct 2026 (PDF)"),
    ("ext/max-languages", 22, "22", "MultiSocial (macko2025multisoc): 22 languages; M4GT-Bench 9, MULTITuDE 11, CEAID 7, DetectRL-X 8", "VERIFIED 11 Oct 2026 (PDFs)"),
    ("ext/wiki-dump-date", "1 May 2022", "1 May 2022", "archive.org item kkwiki-20220501 ('Wikimedia database dump of the Kazakh Wikipedia on May 01, 2022')", "VERIFIED 11 Oct 2026"),
    ("ext/wiki-import-year", 2011, "2011", "en.wikipedia.org/wiki/Kazakh_Wikipedia: rapid growth in 2011 from incorporating Kazakh Encyclopedia material released under CC BY-SA", "VERIFIED 11 Oct 2026 (secondary source)"),
    ("ext/cc100-years", "2018", "2018", "huggingface.co/datasets/statmt/cc100: built from the January-December 2018 Common Crawl snapshots", "VERIFIED 11 Oct 2026"),
    ("ext/encoder-params", 0.28, "0.28", "XLM-R-base 278,885,778 (HF safetensors metadata); mDeBERTa-v3-base 278.8M (released fp32 weights, 1,115,268,200 bytes)", "VERIFIED 9 Oct 2026"),
    ("ext/zs-params", "1.3B--5.1B", "1.3B--5.1B", "HF safetensors metadata: Gemma-4-E2B 5,123,178,051; Qwen3-4B 4,022,468,096; Llama-3.2-3B 3,212,749,824; mGPT 1.3B per model name", "VERIFIED 9 Oct 2026 (mGPT from name)"),
]
for key, v, s, src, status in EXT:
    put(key, v, s, "ext", f"{src} [{status}]")

res = {"inputs": {fn: sha(fn) for fn in INPUTS},
       "frozen_benchmark": frozen["data/eval/eval.jsonl"],
       "analysis": A, "table1": TABLE1, "audit_mt_order": AUD_MT,
       "ots_names": {"ots_fakespot-roberta": ["Fakespot RoBERTa", "fakespot-ai/roberta-base-ai-text-detection-v1"],
                     "ots_openai-roberta": ["OpenAI RoBERTa", "openai-community/roberta-base-openai-detector"],
                     "ots_hc3-roberta": ["HC3 RoBERTa", "Hello-SimpleAI/chatgpt-detector-roberta"],
                     "ots_tmr-roberta": ["TMR RoBERTa", "Oxidane/tmr-ai-text-detector"],
                     "ots_desklib-deberta": ["Desklib DeBERTa", "desklib/ai-text-detector-v1.01"]},
       "gen_names": GEN_NAME, "gen_keys": GK, "cues": CUES,
       "values": V}
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"{OUT}: {len(V)} values ({Counter(x['kind'] for x in V.values())})")
