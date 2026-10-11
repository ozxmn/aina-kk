import bisect, glob, json, math, os, random, re, statistics, subprocess, sys, tempfile
from collections import defaultdict

DIRS = sys.argv[1:]
R = json.load(open("results/results.json"))
V = R["values"]
E = [json.loads(l) for l in open("data/eval/eval.jsonl", encoding="utf-8")]
EI = {r["id"]: r for r in E}
for r in E:
    r["_nw"] = len(r["text"].split())

TEX = ["paper/main.tex", "paper/abstract_body.tex", "paper/findings_body.tex", "paper/fixes_body.tex", "paper/conclusion_body.tex", "paper/appendix_body.tex"]
used = set()
for fn in TEX:
    txt = re.sub(r"(?<!\\)%.*", "", open(fn, encoding="utf-8").read())
    used |= set(re.findall(r"\\res\{([^}]+)\}", txt))
for c in json.load(open("results/table_cells.json")):
    if c["key"]:
        used.add(c["key"])
for alias in ("corpus/n-human", "corpus/n-nonhuman", "corpus/n-gen", "corpus/n-outlets", "corpus/n-edu-hosts"):
    used.add(alias)

meta = next(json.load(open(f"{d}/det_meta.json")) for d in DIRS if os.path.exists(f"{d}/det_meta.json"))
DEVC = set(meta["dev_clusters"])
raw = defaultdict(lambda: defaultdict(list))
seeds = defaultdict(dict)
for d in DIRS:
    for fn in glob.glob(f"{d}/det_*_s*.jsonl") + glob.glob(f"{d}/ots_*.jsonl"):
        base = os.path.basename(fn)[:-6]
        tag = re.sub(r"_s\d+$", "", base)
        for line in open(fn):
            x = json.loads(line)
            if x["p_machine"] == x["p_machine"]:
                raw[tag][x["id"]].append(x["p_machine"])
                seeds[tag].setdefault(base, {})[x["id"]] = x["p_machine"]
    for fn in glob.glob(f"{d}/zs_*.jsonl"):
        sc = os.path.basename(fn)[3:-6]
        for line in open(fn):
            x = json.loads(line)
            for k, sign in (("binoculars", -1), ("obs_fdg", 1), ("perf_fdg", 1)):
                if x[k] == x[k] and not math.isinf(x[k]):
                    raw[f"zs:{sc}:{k}"][x["id"]].append(sign * x[k])
SC = {t: {i: sum(v) / len(v) for i, v in m.items()} for t, m in raw.items()}

def pick(**kw):
    out = []
    for r in E:
        if all((f(r) if callable(f) else r.get(k) == f) for k, f in kw.items()):
            out.append(r["id"])
    return out

Hdev = pick(label="H", variant="orig", src_split="train", cluster=lambda r: r["cluster"] in DEVC)
Htr = pick(label="H", variant="orig", src_split="train")
Hte = pick(label="H", variant="orig", split="test")
Mid = pick(label="M", variant="orig", split="test", heldout=False)
Mho = pick(label="M", variant="orig", heldout=True, src_split="test")
VAR = lambda lab, var: pick(label=lab, variant=var)

def quant(v, q):
    v = sorted(v)
    h = (len(v) - 1) * q
    lo = int(math.floor(h))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (h - lo) * (v[hi] - v[lo])

def share(s, ids, th):
    v = [s[i] >= th for i in ids if i in s]
    return sum(v) / len(v) if v and th is not None else None

def auroc(s, pos, neg):
    p = [s[i] for i in pos if i in s]
    n = sorted(s[i] for i in neg if i in s)
    if len(p) < 5 or len(n) < 5:
        return None
    w = sum(bisect.bisect_left(n, x) + 0.5 * (bisect.bisect_right(n, x) - bisect.bisect_left(n, x)) for x in p)
    return w / (len(p) * len(n))

def pctl(v, q):
    return quant(v, q / 100)

def boot_share(s, ids, th, B=400, seed=0):
    g = {}
    for i in ids:
        g.setdefault(EI[i]["group"], []).append(i)
    keys = list(g)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        rs = [x for k in (rng.choice(keys) for _ in keys) for x in g[k]]
        v = share(s, rs, th)
        if v is not None:
            out.append(v)
    return pctl(out, 2.5), pctl(out, 97.5)

def boot_auc(s, pos, neg, B=300, seed=0):
    g = {}
    for i in pos:
        g.setdefault(EI[i]["group"], []).append((i, 1))
    for i in neg:
        g.setdefault(EI[i]["group"], []).append((i, 0))
    keys = list(g)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        rs = [x for k in (rng.choice(keys) for _ in keys) for x in g[k] if x[0] in s]
        p = [i for i, y in rs if y == 1]
        n = [i for i, y in rs if y == 0]
        if p and n:
            pp = [s[i] for i in p]
            nn = sorted(s[i] for i in n)
            w = sum(bisect.bisect_left(nn, x) + 0.5 * (bisect.bisect_right(nn, x) - bisect.bisect_left(nn, x)) for x in pp)
            out.append(w / (len(pp) * len(nn)))
    return pctl(out, 2.5), pctl(out, 97.5)

def ecdf(t):
    ref = sorted(SC[t][i] for i in Hdev if i in SC[t])
    return {i: bisect.bisect_right(ref, x) / len(ref) for i, x in SC[t].items()}

ZB = "zs:gemma4-e2b:binoculars"
PARTS = {}
for a in ("det_base", "det_aug", "det_auglen", "det_all", "det_alllen"):
    ca, cb = ecdf(a), ecdf(ZB)
    SC[f"combo_{a[4:]}+bino"] = {i: max(ca[i], cb[i]) for i in ca if i in cb}
    PARTS[f"combo_{a[4:]}+bino"] = a

def boot_thr(t, rows, short=None, B=400, seed=0):
    rng = random.Random(seed)
    g = {}
    for i in rows:
        g.setdefault(EI[i]["group"], []).append(i)
    keys = list(g)
    cut = {EI[i]["base_id"]: i for i in VAR("H", f"devtrunc{short}")} if short else None
    out = []
    for _ in range(B):
        ds = [rng.choice(Hdev) for _ in Hdev]
        ts = [x for k in (rng.choice(keys) for _ in keys) for x in g[k]]
        if t in PARTS:
            a = PARTS[t]
            ra, rb = sorted(SC[a][i] for i in ds), sorted(SC[ZB][i] for i in ds)
            score = lambda i: max(bisect.bisect_right(ra, SC[a][i]) / len(ra), bisect.bisect_right(rb, SC[ZB][i]) / len(rb))
        else:
            score = lambda i: SC[t][i]
        th = quant([score(cut[i] if cut else i) for i in ds], 0.99)
        out.append(sum(score(i) >= th for i in ts) / len(ts))
    return pctl(out, 2.5), pctl(out, 97.5)

pct = lambda x: "--" if x is None else f"{100 * x:.1f}"
f3 = lambda x: "--" if x is None else f"{x:.3f}"
GKR = {v: k for k, v in R["gen_keys"].items()}
GENS = sorted({r["generator"] for r in E if r["label"] == "M"})

def cfg_of(part):
    if part.startswith("zs:"):
        _, fam, stat = part.split(":")
        return f"zs:{fam}:{stat.replace('-', '_')}"
    if part.startswith("combo-"):
        return "combo_" + part[6:]
    return "det_" + part.replace("-", "_")

def thr_of(t):
    return quant([SC[t][i] for i in (Htr if t.startswith("zs:") else Hdev) if i in SC[t]], 0.99)

def op_th(t, op):
    return 0.5 if op == "0.5" else thr_of(t)

def devlen_th(t, n):
    v = [SC[t][i] for i in VAR("H", f"devtrunc{n}") if i in SC[t]]
    return quant(v, 0.99) if len(v) >= 50 else None

def det_metric(t, path):
    s = SC[t]
    m = path[0]
    mm = re.fullmatch(r"(fpr|tpr|tpr-heldout)-at-(0\.5|dev)", m)
    if mm and len(path) == 1:
        th = op_th(t, mm.group(2))
        return share(s, {"fpr": Hte, "tpr": Mid, "tpr-heldout": Mho}[mm.group(1)], th), pct
    if m == "fpr-at-dev-ci":
        lo, hi = boot_share(s, Hte, thr_of(t))
        return (lo if path[1] == "lo" else hi), pct
    if m == "fpr-at-dev-ci-thr":
        lo, hi = boot_thr(t, Hte)
        return (lo if path[1] == "lo" else hi), pct
    if m == "auroc":
        return auroc(s, Mid, Hte), f3
    if m == "auroc-heldout":
        return auroc(s, Mho, Hte), f3
    if m == "auroc-ci":
        lo, hi = boot_auc(s, Mid, Hte)
        return (lo if path[1] == "lo" else hi), f3
    if m == "auroc-len120-220":
        return auroc(s, [i for i in Mid if 120 <= EI[i]["_nw"] < 220], [i for i in Hte if 120 <= EI[i]["_nw"] < 220]), f3
    mm = re.fullmatch(r"fpr-by-domain-at-(0\.5|dev)", m)
    if mm:
        return share(s, [i for i in Hte if EI[i]["domain"] == path[1]], op_th(t, mm.group(1))), pct
    mm = re.fullmatch(r"flag-gray-at-(0\.5|dev)", m)
    if mm:
        rows = [i for i in VAR(path[1], "orig") if path[1] == "MT" or EI[i]["src_split"] == "test"]
        return share(s, rows, op_th(t, mm.group(1))), pct
    mm = re.fullmatch(r"flag-by-gen-at-(0\.5|dev)", m)
    if mm:
        g = GKR.get(path[1], path[1])
        return share(s, [i for i in Mid + Mho if EI[i]["generator"] == g], op_th(t, mm.group(1))), pct
    if m == "short":
        n, mt = path[1], path[2]
        Ht, Mt = VAR("H", f"trunc{n}"), VAR("M", f"trunc{n}")
        mm = re.fullmatch(r"(fpr|tpr)-at-(0\.5|dev|devlen)(-ci|-ci-thr)?", mt)
        th = devlen_th(t, n) if mm.group(2) == "devlen" else op_th(t, mm.group(2))
        if mm.group(3) == "-ci-thr":
            lo, hi = boot_thr(t, Ht, short=n)
            return (lo if path[3] == "lo" else hi), pct
        if mm.group(3):
            lo, hi = boot_share(s, Ht, th)
            return (lo if path[3] == "lo" else hi), pct
        return share(s, Ht if mm.group(1) == "fpr" else Mt, th), pct
    if m == "script":
        k = path[1]
        th = thr_of(t)
        if k in ("latin2021", "latin2021canon"):
            return auroc(s, VAR("M", k), VAR("H", k)), f3
        table = {"latin-tpr-at-dev": VAR("M", "latin2021"), "canon-tpr-at-dev": VAR("M", "latin2021canon"), "canon-fpr-at-dev": VAR("H", "latin2021canon"),
                 "latin-fpr-at-dev": VAR("H", "latin2021"), "hg-tpr-at-dev": VAR("M", "homoglyph10"), "hgfold-tpr-at-dev": VAR("M", "homoglyph10fold"),
                 "hg-tpr-orig-at-dev": [EI[i]["base_id"] for i in VAR("M", "homoglyph10")]}
        fo = [i for i in VAR("H", "folded") if EI[i]["split"] == "test"]
        if k == "hghum-n":
            return len(fo), lambda x: f"{x:,}"
        if k == "hghum-raw-at-dev":
            return share(s, [EI[i]["base_id"] for i in fo], th), pct
        if k == "hghum-fold-at-dev":
            return share(s, fo, th), pct
        return share(s, table[k], th), pct
    if m == "seed-range":
        return None, None
    raise KeyError(path)

def seed_span(t, metric):
    vals = []
    for base, sd in seeds[t].items():
        th = quant([sd[i] for i in Hdev if i in sd], 0.99)
        rows = {"tpr-at-dev": Mid, "fpr-at-dev": Hte, "tpr-heldout-at-dev": Mho, "fpr50-at-dev": VAR("H", "trunc50")}[metric]
        vals.append(share(sd, rows, th))
    lo, hi = pct(min(vals)), pct(max(vals))
    return lo if lo == hi else f"{lo}--{hi}"

def nwords(ids):
    return [EI[i]["_nw"] for i in ids]

O = [r for r in E if r["variant"] == "orig"]

def median(v):
    return statistics.median(v)

def compute(k):
    p = k.split("/")
    if p[0] == "det" and p[1] == "n":
        n = {"H-dev": len(Hdev), "H-test": len(Hte), "M-indist": len(Mid), "M-heldout-topicdisjoint": len(Mho), "H-train-zs": len(Htr), "dev-clusters": len(DEVC)}[p[2]]
        return f"{n:,}"
    if p[0] == "det" and p[1] == "complement":
        a = "det_" + p[2]
        ta, tb = thr_of(a), thr_of(ZB)
        kz = [i for i in Mho if "KazLLM" in EI[i]["generator"] or "Sherkala" in EI[i]["generator"]]
        missed = [i for i in kz if i in SC[a] and i in SC[ZB] and SC[a][i] < ta]
        if p[3] == "n-missed":
            return f"{len(missed):,}"
        return pct(sum(SC[ZB][i] >= tb for i in missed) / len(missed))
    if p[0] == "det" and p[1] == "logo":
        g = GKR.get(p[2], p[2])
        t = "det_logo-" + g
        Mg = [i for i in Mid + Mho if EI[i]["generator"] == g]
        m = p[3]
        if m == "auroc":
            return f3(auroc(SC[t], Mg, Hte))
        if m == "auroc-when-trained":
            return f3(auroc(SC["det_auglen"], Mg, Hte))
        if m == "tpr-at-dev":
            return pct(share(SC[t], Mg, thr_of(t)))
        if m == "tpr-at-dev-when-trained":
            return pct(share(SC["det_auglen"], Mg, thr_of("det_auglen")))
    if k == "ots/n":
        return WORDS[sum(1 for t in SC if t.startswith("ots_"))]
    if k == "det/n-combos":
        return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}[len(PARTS)]
    if p[0] == "det":
        t = cfg_of(p[1])
        val, fmt = det_metric(t, p[2:])
        return fmt(val)
    if p[0] == "ots":
        t = "ots_" + p[1]
        val, fmt = det_metric(t, p[2:])
        return fmt(val)
    if p[0] == "seedspan":
        return seed_span("det_" + p[1].replace("-", "_"), p[2])
    if p[0] == "flag":
        t = "det_" + p[1]
        glob_ = [g for g in GENS if "KazLLM" not in g and "Sherkala" not in g]
        th = thr_of(t)
        return pct(min(share(SC[t], [i for i in Mid + Mho if EI[i]["generator"] == g], th) for g in glob_))
    if p[0] in ("ppv", "fdr", "falseflags"):
        name, op = p[1], p[2]
        t = {"combo-auglen+bino": "combo_auglen+bino", "base": "det_base"}.get(name, "ots_" + name[4:] if name.startswith("ots-") else name)
        th = op_th(t, op)
        fpr = share(SC[t], Hte, th)
        if p[0] == "falseflags":
            return f"{round(1000 * fpr):d}"
        a, b = share(SC[t], Mid, th), share(SC[t], Mho, th)
        tpr = (a * len(Mid) + b * len(Mho)) / (len(Mid) + len(Mho))
        d = tpr * 0.05 + fpr * 0.95
        pv = tpr * 0.05 / d if d > 0 else None
        return pct(pv) if p[0] == "ppv" else ("--" if pv is None else f"{100 * (1 - pv):.0f}")
    if p[0] == "ots":
        pass
    if k in ("ots/maxfpr", "ots/max-essay-fpr", "ots/max-tpr-dev"):
        pass
    if p[0] == "zs":
        rows = ["zs:gemma4-e2b:binoculars", "zs:gemma4-e2b:obs_fdg", "zs:llama3.2-3b:binoculars", "zs:llama3.2-3b:obs_fdg", "zs:qwen3-4b:binoculars",
                "zs:qwen3-4b:obs_fdg", "zs:mgpt-1.3b:obs_fdg", "zs:mgpt-1.3b:perf_fdg"]
        tp = sorted((share(SC[t], Mid, thr_of(t)) for t in rows), reverse=True)
        fp = [share(SC[t], Hte, thr_of(t)) for t in rows]
        return pct({"best-tpr": tp[0], "second-tpr": tp[1], "max-fpr-dev": max(fp), "min-fpr-dev": min(fp)}[p[1]])
    if p[0] == "len":
        H = [r["_nw"] for r in O if r["label"] == "H"]
        M = [r["_nw"] for r in O if r["label"] == "M"]
        G = {"sherkala": "Sherkala", "kazllm": "KazLLM"}
        if p[1] == "h-median":
            return str(int(median(H)))
        if p[1] == "m-median":
            return str(int(median(M)))
        if p[1] == "h-under120":
            return pct(sum(x < 120 for x in H) / len(H))
        if p[1] == "m-under120":
            return pct(sum(x < 120 for x in M) / len(M))
        g, what = p[1].split("-", 1)
        v = [r["_nw"] for r in O if r["label"] == "M" and G[g] in r["generator"]]
        return pct(sum(x < 120 for x in v) / len(v)) if what == "under120" else str(int(median(v)))
    W = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    if k == "corpus/n-domains":
        return W[len({r['domain'] for r in O if r['label'] == 'H'})]
    if k == "det/n-combos":
        return W[len(PARTS)]
    if p[0] == "devrep":
        return devrep(p[1:])
    if p[0] == "val":
        return val_check(p[1:])
    if k == "corpus/n-train-gens":
        return W[len({r['generator'] for r in O if r['label'] == 'M' and r['split'] == 'train'})]
    if k == "corpus/n-heldout-gens":
        return W[len({r['generator'] for r in O if r.get('heldout')})]
    if k == "corpus/n-heldout-gemma":
        return W[len({r['generator'] for r in O if r.get('heldout') and 'gemma-4' in r['generator']})]
    if k == "corpus/n-zs-families":
        return W[len({t.split(':')[1] for t in SC if t.startswith('zs:')})]
    if k == "corpus/heldout-names":
        names = R["gen_names"]
        return ", ".join(names.get(g, g) for g in sorted({r["generator"] for r in O if r.get("heldout")})).replace("_", "\\_")
    if k == "compute/ft-run-minutes":
        ts = []
        for f_ in glob.glob("kaggle/out_det_final_*/*.log"):
            ts += [int(x) for x in re.findall(r'"data":"done [a-z0-9]+ \d ([0-9]+)', open(f_).read())]
        return W[int(round(statistics.median(ts) / 60))]
    if p[0] == "corpus":
        if p[1] == "n-human":
            return f"{sum(r['label'] == 'H' for r in O):,}"
        if p[1] == "n-nonhuman":
            return f"{sum(r['label'] != 'H' for r in O):,}"
        if p[1] == "n-gen":
            return str(len({r['generator'] for r in O if r['label'] != 'H'}))
        if p[1] == "variant":
            return f"{sum(1 for r in E if r['label'] == p[2] and r['variant'] == p[3]):,}"
        if p[1] == "gen":
            g = GKR.get(p[2], p[2])
            return str(sum(1 for r in O if r["generator"] == g and r["label"] == p[3]))
        if p[1] in ("n-heldout-m", "n-heldout-mix"):
            lab = {"n-heldout-m": "M", "n-heldout-mix": "MIX"}[p[1]]
            return f"{sum(1 for r in O if r.get('heldout') and r['label'] == lab):,}"
        if p[1] == "n-heldout-rows":
            return f"{sum(1 for r in O if r.get('heldout')):,}"
        if p[1] == "n-domains-n":
            return str(len({r['domain'] for r in O if r['label'] == 'H'}))
        if p[1] in ("n-outlets", "n-edu-hosts"):
            dom = "news" if p[1] == "n-outlets" else "essay"
            src = set()
            for fn in ("data/human/human_clean.jsonl", "data/human/essay_clean.jsonl"):
                for l in open(fn, encoding="utf-8"):
                    h = json.loads(l)
                    if h["domain"] == dom:
                        src.add(h.get("source"))
            return str(len(src))
        return None
    if p[0] == "aud" and p[1] == "post":
        return aud_post(p[2:])
    if p[0] == "aud":
        HA = {json.loads(l)["id"]: json.loads(l) for l in open("results/human_audit.jsonl", encoding="utf-8")}
        LA = {json.loads(l)["id"]: json.loads(l) for l in open("results/llm_audit.jsonl", encoding="utf-8")}
        both = [i for i in HA if i in LA and HA[i]["ok"] is not None]
        rate = lambda D, kind: sum(D[i]["ok"] for i in D if D[i]["kind"] == kind) / sum(1 for i in D if D[i]["kind"] == kind)
        if p[1] == "n-human-judged":
            return str(len(HA))
        if p[1] == "n-llm":
            return str(len(LA))
        if p[1] == "n-both":
            return str(len(both))
        if p[1] == "human-valid-h":
            return pct(rate(HA, "human"))
        if p[1] == "human-valid-m":
            return pct(rate(HA, "machine"))
        if p[1] == "llm-valid-h":
            return pct(rate(LA, "human"))
        if p[1] == "llm-valid-m":
            return pct(rate(LA, "machine"))
        if p[1] == "raw":
            return pct(sum(bool(HA[i]["ok"]) == bool(LA[i]["ok"]) for i in both) / len(both))
        if p[1] == "kappa":
            h = [int(bool(HA[i]["ok"])) for i in both]
            a = [int(bool(LA[i]["ok"])) for i in both]
            n = len(h)
            po = sum(x == y for x, y in zip(h, a)) / n
            pe = (sum(h) / n) * (sum(a) / n) + (1 - sum(h) / n) * (1 - sum(a) / n)
            return f"{(po - pe) / (1 - pe):.2f}"
        if p[1] == "n-disagree":
            return str(sum(bool(HA[i]["ok"]) != bool(LA[i]["ok"]) for i in both))
        if p[1] == "n-disagree-mt":
            return str(sum(1 for i in both if HA[i]["kind"] == "machine" and HA[i]["reason"] == "looks_mt" and LA[i]["ok"]))
        if p[1] == "mt":
            items = {x["id"]: x for x in json.load(open("audit/items.json", encoding="utf-8"))}
            g = GKR.get(p[2], p[2])
            gen = lambda i: items[i]["generator"].split(":")[-1].split("/")[-1]
            if p[3] == "n":
                return str(sum(1 for i in HA if HA[i]["kind"] == "machine" and HA[i]["reason"] == "looks_mt" and gen(i) == g))
            return str(sum(1 for i in HA if HA[i]["kind"] == "machine" and gen(i) == g))
    if p[0] == "logo" and p[1] == "min-global-out":
        out = []
        for g in GENS:
            if "KazLLM" in g or "Sherkala" in g or f"det_logo-{g}" not in SC:
                continue
            t = "det_logo-" + g
            out.append(share(SC[t], [i for i in Mid + Mho if EI[i]["generator"] == g], thr_of(t)))
        return pct(min(out))
    if p[0] == "canon":
        import difflib
        rows = [r for r in E if r["variant"] == "latin2021canon"]
        if p[1] == "n":
            return f"{len(rows):,}"
        if p[1] == "charsim":
            return pct(sum(difflib.SequenceMatcher(None, EI[r["base_id"]]["text"], r["text"], autojunk=False).ratio() for r in rows) / len(rows))
        if p[1] == "word-exact":
            same = tot = 0
            for r in rows:
                a, b = EI[r["base_id"]]["text"].split(), r["text"].split()
                if len(a) == len(b):
                    same += sum(x == y for x, y in zip(a, b)); tot += len(a)
            return pct(same / tot)
    if p[0] == "overlap":
        G = {"sherkala": "Sherkala", "kazllm": "KazLLM"}[p[1]]
        hum = defaultdict(list)
        for r in O:
            if r["label"] == "H":
                w = re.findall(r"\w+", r["text"].lower())
                hum[r["group"]].append({tuple(w[i:i + 6]) for i in range(len(w) - 5)})
        vals = []
        for r in O:
            if r["label"] == "M" and G in r["generator"]:
                w = re.findall(r"\w+", r["text"].lower())
                m = {tuple(w[i:i + 6]) for i in range(len(w) - 5)}
                vals.append(max((len(m & h) / max(1, len(m)) for h in hum.get(r["group"], [])), default=0.0))
        return pct(sum(vals) / len(vals)) if p[2] == "mean" else pct(quant(vals, 0.95))
    if p[0] == "compute":
        if p[1] == "gpu-hours":
            tot = 0.0
            for f_ in glob.glob("kaggle/out*/*.log"):
                ts = [float(x) for x in re.findall(r'"time":([0-9.]+)', open(f_).read())]
                if ts:
                    tot += max(ts) / 3600
            return str(int(round(tot)))
        return None
    if p[0] == "lenratio":
        tw = {}
        for fn in glob.glob("data/machine/*.tasks.jsonl") + glob.glob("kaggle/gen_inputs/*.tasks.jsonl"):
            for l in open(fn, encoding="utf-8"):
                t = json.loads(l)
                if t.get("target_words"):
                    tw[t["task_id"]] = t["target_words"]
        ratio = defaultdict(list)
        for l in open("data/machine/machine_pool.jsonl", encoding="utf-8"):
            q = json.loads(l)
            r = EI.get(q["id"])
            if r and r["variant"] == "orig" and r["label"] == "M" and q["task_id"] in tw:
                ratio[r["generator"]].append(r["_nw"] / tw[q["task_id"]])
        med = {g: median(v) for g, v in ratio.items()}
        f0 = lambda x: f"{100 * x:.0f}"
        if p[1] == "min":
            return f0(min(med.values()))
        if p[1] == "max":
            return f0(max(med.values()))
        return f0(med[GKR.get(p[1], p[1])])
    if p[0] == "qc":
        Q = json.load(open("results/qc_machine_log.json"))["drops"]
        if p[1] == "mixed-script-drops":
            return str(sum(v.get("mixed_script", 0) for v in Q.values()))
        return str(sum(v.get("mixed_script", 0) for g, v in Q.items() if "3.5-flash-lite" in g))
    return None

WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

DEV_ROWS = {"base05": ("det_base", "0.5"), "base": ("det_base", "cal"), "aug": ("det_aug", "cal"), "auglen": ("det_auglen", "cal"),
            "auglen-lenthr": ("det_auglen", "len"), "combo": ("combo:det_auglen", "len"), "kznoaug": ("det_kznoaug", "cal"), "kz": ("det_kz", "cal"),
            "kzlen": ("det_kzlen", "len"), "combo-base": ("combo:det_base", "len"), "combo-aug": ("combo:det_aug", "len"), "combo-auglen": ("combo:det_auglen", "len")}
DM = [r["id"] for r in E if r["label"] == "M" and r["variant"] == "orig" and r["src_split"] == "train" and r["cluster"] in DEVC]
D50 = [i for i in VAR("H", "devtrunc50") if EI[i]["src_split"] == "train" and EI[i]["cluster"] in DEVC]
_drcache = {}

def devrep_row(name):
    if name in _drcache:
        return _drcache[name]
    tag, mode = DEV_ROWS[name]
    hits = defaultdict(list)
    for c in sorted(DEVC):
        fit = [i for i in Hdev if EI[i]["cluster"] != c]
        if tag.startswith("combo:"):
            a = tag[6:]
            ra, rb = sorted(SC[a][i] for i in fit), sorted(SC[ZB][i] for i in fit)
            f = lambda i, a=a, ra=ra, rb=rb: max(bisect.bisect_right(ra, SC[a][i]) / len(ra), bisect.bisect_right(rb, SC[ZB][i]) / len(rb))
        else:
            f = lambda i, tag=tag: SC[tag][i]
        if mode == "0.5":
            th = th50 = 0.5
        else:
            th = quant([f(i) for i in fit], 0.99)
            th50 = quant([f(i) for i in D50 if EI[i]["cluster"] != c], 0.99) if mode == "len" else th
        for i in Hdev:
            if EI[i]["cluster"] == c:
                hits["fpr"].append(f(i) >= th)
                if EI[i]["domain"] == "web":
                    hits["fpr-web"].append(f(i) >= th)
        for i in D50:
            if EI[i]["cluster"] == c:
                hits["fpr50"].append(f(i) >= th50)
        for i in DM:
            if EI[i]["cluster"] == c:
                g = EI[i]["generator"]
                kk = "tpr-kazllm" if "KazLLM" in g else "tpr-sherkala" if "Sherkala" in g else "tpr-gemma" if EI[i]["heldout"] else "tpr-indist"
                hits[kk].append(f(i) >= th)
    _drcache[name] = hits
    return hits

def devrep(p):
    if p[0] == "n-h":
        return f"{len(Hdev):,}"
    if p[0] == "n-m":
        return f"{len(DM):,}"
    if p[0] == "n-fit-clusters":
        return WORDS[len(DEVC) - 1]
    if p[0] == "n":
        return str(len(devrep_row("combo")[p[1]]))
    h = devrep_row(p[0])[p[1]]
    return pct(sum(h) / len(h))

def val_check(p):
    Hh = [r for r in O if r["label"] == "H"]
    if p[0] == "split":
        trc = {r["cluster"] for r in Hh if r["split"] == "train"}
        tec = {r["cluster"] for r in Hh if r["split"] == "test"}
        ess = {r["cluster"] for r in Hh if r["domain"] == "essay"}
        if p[1] == "train-clusters":
            return str(len(trc))
        if p[1] == "test-clusters":
            return str(len(tec))
        if p[1] == "shared-clusters":
            return str(len(trc & tec))
        if p[1] == "essay-test-clusters":
            return WORDS[len(tec & ess)]
        if p[1] == "essay-train-clusters":
            return WORDS[len(trc & ess)]
        if p[1].startswith("dev-"):
            return str(sum(1 for i in Hdev if EI[i]["domain"] == p[1][4:]))
        return None
    if p[0] == "digits":
        has = lambda i: any(ch.isdigit() and ch in "0123456789" for ch in EI[i]["text"])
        H0 = [i for i in Hte if not has(i)]
        H1 = [i for i in Hte if has(i)]
        if p[1] == "h":
            return f"{100 * len(H1) / len(Hte):.0f}"
        if p[1] == "m":
            return f"{100 * sum(has(i) for i in Mid) / len(Mid):.0f}"
        if p[1] == "n-h-nodigit":
            return f"{len(H0):,}"
        if p[1] == "n-h-nodigit-essay":
            return f"{sum(EI[i]['domain'] == 'essay' for i in H0):,}"
        t = {"base": "det_base", "combo": "combo_auglen+bino"}[p[1]]
        th = thr_of(t)
        if p[2] == "auroc-nodigit":
            return f3(auroc(SC[t], [i for i in Mid if not has(i)], H0))
        if p[2] == "flagged-nodigit":
            return str(sum(SC[t][i] >= th for i in H0))
        return pct(share(SC[t], H0 if p[2] == "fpr-nodigit" else H1, th))
    if p[0] == "family":
        if p[1] == "n-kz":
            return str(sum(1 for i in Mho if "KazLLM" in EI[i]["generator"] or "Sherkala" in EI[i]["generator"]))
        if p[1] == "n-gemma":
            return str(sum(1 for i in Mho if "gemma-4" in EI[i]["generator"]))
        t = {"auglen": "det_auglen", "combo": "combo_auglen+bino"}[p[1]]
        rows = [i for i in Mho if ("gemma-4" in EI[i]["generator"]) == (p[2] == "gemma")]
        return pct(share(SC[t], rows, thr_of(t)))
    if p[0] == "latin":
        ns = [t for t in SC if t in ("det_base", "det_xlmr", "det_kznoaug") or t.startswith(("zs:", "ots_"))]
        rates = {t: (share(SC[t], VAR("M", "latin2021"), thr_of(t)), share(SC[t], VAR("H", "latin2021"), thr_of(t))) for t in ns}
        low = [v for v in rates.values() if v[1] < 0.10]
        high = [v for v in rates.values() if v[1] >= 0.10]
        return {"n": str(len(rates)), "n-low": str(len(low)), "max-tpr-low": pct(max(v[0] for v in low)), "n-high": WORDS[len(high)],
                "min-fpr-high": f"{100 * min(v[1] for v in high):.0f}", "max-fpr-high": f"{100 * max(v[1] for v in rates.values()):.0f}"}[p[1]]
    if p[0] == "pools":
        ind = {r["generator"] for r in O if r["label"] == "M" and not r["heldout"]}
        kzg = {r["generator"] for r in O if r.get("heldout") and ("KazLLM" in r["generator"] or "Sherkala" in r["generator"])}
        pool = lambda gens: [r for r in O if r["src_split"] == "train" and r["cluster"] not in DEVC and (r["label"] == "H" or (r["label"] == "M" and r["generator"] in gens))]
        P1, P2 = pool(ind), pool(ind | kzg)
        return str({"main-heldout-rows": sum(r["heldout"] for r in P1), "main-test-rows": sum(r["split"] == "test" and r["label"] == "H" for r in P1),
                    "kz-gemma-rows": sum("gemma-4" in r["generator"] for r in P2), "dev-rows-any": sum(r["cluster"] in DEVC for r in P1 + P2),
                    "kz-rows": sum(r["generator"] in kzg for r in P2)}[p[1]])
    if p[0] == "dates":
        ids = {r["id"] for r in Hh}
        v = []
        for fn in ("data/human/human_clean.jsonl", "data/human/essay_clean.jsonl"):
            for l in open(fn, encoding="utf-8"):
                h = json.loads(l)
                if h["id"] in ids and h["domain"] == p[1]:
                    v.append(h.get("rev_timestamp") or h.get("crawl_date"))
        if p[2] == "n":
            return f"{len(v):,}"
        if p[2] == "dated":
            return f"{sum(x is not None for x in v):,}"
        y, mo, d = max(x for x in v if x)[:10].split("-")
        return f"{int(d)} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(mo) - 1]} {y}"
    return None

def aud_post(p):
    items = json.load(open("audit/postfilter/items.json", encoding="utf-8"))
    J = {}
    for l in open("audit/postfilter/judgments.jsonl", encoding="utf-8"):
        x = json.loads(l)
        J[x["id"]] = x
    hum = [i for i in items if i["kind"] == "human"]
    mac = [i for i in items if i["kind"] == "machine"]
    k = p[0]
    if k == "n":
        return str(len(items))
    if k == "n-h":
        return str(len(hum))
    if k == "n-m":
        return str(len(mac))
    if k == "n-h-per-domain":
        c = {}
        for i in hum:
            c[i["stratum"]] = c.get(i["stratum"], 0) + 1
        return str(c["news"]) if len(set(c.values())) == 1 else None
    if k == "n-m-per-gen":
        c = {}
        for i in mac:
            c[i["stratum"]] = c.get(i["stratum"], 0) + 1
        return str(next(iter(c.values()))) if len(set(c.values())) == 1 else None
    if k == "overlap-first":
        old = {x["id"] for x in json.load(open("audit/items.json", encoding="utf-8"))}
        return str(sum(i["id"] in old for i in items))
    if k == "h-valid":
        return str(sum(J[i["id"]]["ok"] for i in hum))
    if k in ("h-valid-w", "h-valid-w-ci"):
        size = {}
        for r in O:
            if r["label"] == "H":
                size[r["domain"]] = size.get(r["domain"], 0) + 1
        by = {}
        for i in hum:
            by.setdefault(EI[i["id"]]["domain"], []).append(int(J[i["id"]]["ok"]))
        Wt = sum(size[d] for d in by)
        ws = lambda h: sum(size[d] * sum(v) / len(v) for d, v in h.items()) / Wt
        if k == "h-valid-w":
            return f"{100 * ws(by):.0f}"
        rng = random.Random(0)
        bt = sorted(ws({d: [rng.choice(v) for _ in v] for d, v in by.items()}) for _ in range(2000))
        return f"{100 * (bt[49] if p[1] == 'lo' else bt[1949]):.0f}"
    reasons = lambda its: [J[i["id"]]["reason"] for i in its if not J[i["id"]]["ok"]]
    if k == "h-broken":
        return str(reasons(hum).count("broken"))
    if k == "h-mt":
        return str(reasons(hum).count("looks_mt"))
    if k == "h-other":
        return str(sum(1 for x in reasons(hum) if x not in ("broken", "looks_mt")))
    if k == "m-broken":
        return WORDS[reasons(mac).count("broken")]
    if k == "m-mt":
        return str(reasons(mac).count("looks_mt"))
    valid = [i["id"] for i in hum if J[i["id"]]["ok"]]
    invalid = [i["id"] for i in hum if not J[i["id"]]["ok"]]
    if k == "n-valid":
        return str(len(valid))
    if k == "n-invalid":
        return str(len(invalid))
    t = "ots_" + k
    return str(sum(SC[t][i] >= 0.5 for i in (valid if p[1] == "valid" else invalid)))

AGG = {"ots/maxfpr": lambda: pct(max(share(SC[t], Hte, 0.5) for t in SC if t.startswith("ots_"))),
       "ots/max-essay-fpr": lambda: pct(max(share(SC[t], [i for i in Hte if EI[i]["domain"] == "essay"], 0.5) for t in SC if t.startswith("ots_"))),
       "ots/max-tpr-dev": lambda: pct(max(share(SC[t], Mid, thr_of(t)) for t in SC if t.startswith("ots_")))}

status = {}
bad = []
rerun_keys = [k for k in used if k.startswith(("cue/", "ling/"))]
VAL_RERUN = ("val/surface/", "val/thr/", "val/templates/", "val/split/n-neardup")
val_rerun = [k for k in used if k.startswith(VAL_RERUN)]
for k in sorted(used):
    if k.startswith("x/"):
        status[k] = "EXTRA"; continue
    if k not in V:
        status[k] = "UNDEFINED"; bad.append((k, "undefined", None)); continue
    kind = V[k]["kind"]
    if kind in ("const", "ext"):
        status[k] = kind.upper()
        continue
    if k in rerun_keys or k in val_rerun:
        continue
    try:
        got = AGG[k]() if k in AGG else compute(k)
    except Exception as ex:
        got = None
        err = f"{type(ex).__name__}: {ex}"
        if os.environ.get("VERIFY_DEBUG"):
            print("  ERROR", k, err)
    if got is None:
        status[k] = "NOT-RECOMPUTED"
        continue
    if got == V[k]["s"]:
        status[k] = "MATCH"
    else:
        status[k] = "MISMATCH"; bad.append((k, V[k]["s"], got))
if rerun_keys:
    tmp = tempfile.mkdtemp()
    ok = True
    for script, out in (("scripts/surface_cues.py", "results/surface_cues.json"), ("scripts/ling.py", "results/linguistic.json")):
        before = open(out).read()
        subprocess.run([sys.executable, script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        after = open(out).read()
        if json.loads(before) != json.loads(after):
            ok = False
            open(os.path.join(tmp, os.path.basename(out)), "w").write(after)
            open(out, "w").write(before)
    for k in rerun_keys:
        status[k] = "RERUN-MATCH" if ok else "RERUN-DIFFERS"
        if not ok:
            bad.append((k, V[k]["s"], "rerun differs"))
if val_rerun:
    before = open("results/validity.json").read()
    subprocess.run([sys.executable, "scripts/validity.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    after = open("results/validity.json").read()
    same = json.loads(before) == json.loads(after)
    if not same:
        open("results/validity.json", "w").write(before)
    for k in val_rerun:
        status[k] = "RERUN-MATCH" if same else "RERUN-DIFFERS"
        if not same:
            bad.append((k, V[k]["s"], "rerun differs"))
json.dump(status, open("results/verify_all.json", "w"), indent=0)
from collections import Counter
print(f"{len(used)} values used by the paper: {dict(Counter(status.values()))}")
for b in bad[:60]:
    print("  MISMATCH", b)
nr = [k for k, v in status.items() if v == "NOT-RECOMPUTED"]
if nr:
    print("  not recomputed:", nr[:40])
sys.exit(1 if bad else 0)
