import json, sys
from scipy.stats import beta

X = json.load(open(sys.argv[1])); D = json.load(open(sys.argv[2])); RATES = json.load(open(sys.argv[3])); OUT = sys.argv[4]
V = {}
def put(key, v, s, src, kind="data"):
    V[key] = {"v": v, "s": s, "kind": kind, "src": src}
pc = lambda x: f"{100 * x:.1f}"
def pcs(key, x, src): put(key, x, pc(x), src)

T4 = X["table4_deployed"]
for tag, t in (("auglen", "det_auglen"), ("combo", "combo:det_auglen"), ("kzlen", "det_kzlen"), ("combo-aug", "combo:det_aug")):
    e = T4[t]
    for k in ("fpr", "fpr_web", "fpr_essay", "tpr", "tpr_heldout", "kazllm", "sherkala", "kz", "gemma4", "latin_canon", "latin_canon_fpr", "hgfold", "fpr50", "tpr50", "ppv5"):
        pcs(f"x/dep/{tag}/{k.replace('_', '-')}", e[k], f"extra.json:table4_deployed/{t}/{k}")
    for g, x in e["gray"].items():
        pcs(f"x/dep/{tag}/gray/{g}", x, f"extra.json:table4_deployed/{t}/gray/{g}")
for tag, t in (("auglen", "det_auglen"), ("combo", "combo:det_auglen"), ("kzlen", "det_kzlen")):
    c = X["ci_deployed"][t]
    pcs(f"x/dep/{tag}/fpr-ci/lo", c["fpr_ci_thr"][0], f"extra.json:ci_deployed/{t}/fpr_ci_thr[0]")
    pcs(f"x/dep/{tag}/fpr-ci/hi", c["fpr_ci_thr"][1], f"extra.json:ci_deployed/{t}/fpr_ci_thr[1]")
    for g in ("kazllm", "sherkala", "kz", "gemma4"):
        e = c[g]
        put(f"x/dep/{tag}/{g}/k", e["k"], str(e["k"]), f"extra.json:ci_deployed/{t}/{g}/k")
        put(f"x/dep/{tag}/{g}/n", e["n"], str(e["n"]), f"extra.json:ci_deployed/{t}/{g}/n")
        for side, i in (("lo", 0), ("hi", 1)):
            pcs(f"x/dep/{tag}/{g}/cp/{side}", e["cp"][i], f"extra.json:ci_deployed/{t}/{g}/cp[{i}] (exact Clopper-Pearson)")
            pcs(f"x/dep/{tag}/{g}/boot/{side}", e["boot"][i], f"extra.json:ci_deployed/{t}/{g}/boot[{i}] (bootstrap, thresholds re-fit)")
c = X["deployed"]["combo:det_aug"]["fpr_ci_thr"]["deployed"]
pcs("x/dep/combo-aug/fpr-ci/lo", c[0], "extra.json:deployed/combo:det_aug/fpr_ci_thr/deployed[0]")
pcs("x/dep/combo-aug/fpr-ci/hi", c[1], "extra.json:deployed/combo:det_aug/fpr_ci_thr/deployed[1]")
for tag, t in (("base", "det_base"), ("bino", "zs:gemma4-e2b:binoculars")):
    for g in ("kazllm", "sherkala", "kz", "gemma4"):
        e = X["ci"][t][g]
        put(f"x/ci/{tag}/{g}/k", e["k"], str(e["k"]), f"extra.json:ci/{t}/{g}/k")
        put(f"x/ci/{tag}/{g}/n", e["n"], str(e["n"]), f"extra.json:ci/{t}/{g}/n")
        for side, i in (("lo", 0), ("hi", 1)):
            pcs(f"x/ci/{tag}/{g}/cp/{side}", e["cp"][i], f"extra.json:ci/{t}/{g}/cp[{i}]")
            if "boot" in e: pcs(f"x/ci/{tag}/{g}/boot/{side}", e["boot"][i], f"extra.json:ci/{t}/{g}/boot[{i}]")
for tag, key in (("base", "det_base vs combo:det_base"), ("aug", "det_aug vs combo:det_aug"), ("auglen", "det_auglen vs combo:det_auglen"), ("augcombo", "combo:det_aug vs combo:det_auglen")):
    m = X["mcnemar_kz_deployed"][key]
    put(f"x/mc/{tag}/only-first", m["only_first"], str(m["only_first"]), f"extra.json:mcnemar_kz_deployed/{key}/only_first")
    put(f"x/mc/{tag}/only-second", m["only_second"], str(m["only_second"]), f"extra.json:mcnemar_kz_deployed/{key}/only_second")
    put(f"x/mc/{tag}/p", m["p_exact"], f"{m['p_exact']:.2f}", f"extra.json:mcnemar_kz_deployed/{key}/p_exact (exact binomial)")
Z = X["zs_all"]
put("x/zs/n-stats", len(Z), str(len(Z)), "extra.json:zs_all (statistics stored by the scoring kernels, 4 families x 11)")
ranked = sorted(Z.items(), key=lambda kv: -kv[1]["tpr"])
pcs("x/zs/second-tpr", ranked[1][1]["tpr"], f"extra.json:zs_all/{ranked[1][0]}/tpr (second-highest in-distribution TPR of all statistics)")
pcs("x/zs/second-fpr", ranked[1][1]["fpr"], f"extra.json:zs_all/{ranked[1][0]}/fpr")
put("x/zs/second-name", ranked[1][0], "Gemma-4-E2B performer log-rank" if ranked[1][0] == "gemma4-e2b:perf_logrank" else ranked[1][0], "extra.json:zs_all ranking", kind="name")
pcs("x/zs/max-fpr", max(v["fpr"] for v in Z.values()), "extra.json:zs_all max fpr")
pcs("x/zs/min-fpr", min(v["fpr"] for v in Z.values()), "extra.json:zs_all min fpr")
for k, v in Z.items():
    for m in ("tpr", "fpr", "tpr_heldout"):
        pcs(f"x/zs/{k}/{m.replace('_', '-')}", v[m], f"extra.json:zs_all/{k}/{m}")
    put(f"x/zs/{k}/auroc", v["auroc"], f"{v['auroc']:.3f}", f"extra.json:zs_all/{k}/auroc")
for tag, row in (("auglen", "auglen-deploy"), ("combo", "combo-deploy"), ("kzlen", "kzlen-deploy")):
    for m, x in D["rows"][row].items():
        if m != "n": pcs(f"x/devdep/{tag}/{m.replace('_', '-')}", x, f"dev_replication_v2.json:rows/{row}/{m}")
for a in ("det_base", "det_aug", "det_auglen"):
    for m, x in D["combos_deploy"][a].items():
        if m != "n": pcs(f"x/devdep/combo-{a[4:]}/{m.replace('_', '-')}", x, f"dev_replication_v2.json:combos_deploy/{a}/{m}")
    for m in ("tpr_kazllm", "tpr_sherkala"):
        k = round(D["combos_deploy"][a][m] * D["combos_deploy"][a]["n"][m])
        put(f"x/devdep/combo-{a[4:]}/{m.replace('_', '-')}/k", k, str(k), f"dev_replication_v2.json:combos_deploy/{a}/{m} x n")
for row in ("auglen-deploy",):
    for m in ("tpr_kazllm", "tpr_sherkala"):
        k = round(D["rows"][row][m] * D["rows"][row]["n"][m])
        put(f"x/devdep/auglen/{m.replace('_', '-')}/k", k, str(k), f"dev_replication_v2.json:rows/{row}/{m} x n")
byname = {r["name"]: r for r in RATES["rows"]}
for tag, name in (("tpr", "Machine text, training generators"), ("tpr-ho", "Machine text, unseen generators"),
                  ("kazllm", "Machine text, LLama-3.1-KazLLM-1.0-8B"), ("sherkala", "Machine text, Llama-3.1-Sherkala-8B-Chat"),
                  ("fpr", "Human passages, all domains")):
    pcs(f"x/release/{tag}", byname[name]["rate"], f"rates.json (release_model): {name}")
kz = (byname["Machine text, LLama-3.1-KazLLM-1.0-8B"]["rate"] * 85 + byname["Machine text, Llama-3.1-Sherkala-8B-Chat"]["rate"] * 75) / 160
pcs("x/release/kz", kz, "rates.json: KazLLM (85) and Sherkala (75) pooled")
M = X["misc"]
pcs("x/hg/all", M["homoglyph_share_all_humans"], "extra.json:misc/homoglyph_share_all_humans (3,811 benchmark passages)")
pcs("x/hg/test", M["homoglyph_share_test_humans"], "extra.json:misc/homoglyph_share_test_humans")
put("x/kz/under110", M["short_full_passages"]["kz_under110"], str(M["short_full_passages"]["kz_under110"]), "extra.json:misc/short_full_passages/kz_under110")
put("x/human/under110", M["short_full_passages"]["human_test_under110"], str(M["short_full_passages"]["human_test_under110"]), "extra.json:misc/short_full_passages/human_test_under110")
e = T4["combo:det_auglen"]
t = (e["tpr"] * 1274 + e["tpr_heldout"] * 442) / 1716
hi = X["ci_deployed"]["combo:det_auglen"]["fpr_ci_thr"][1]
put("x/dep/combo/ppv5-at-hi", 0.05 * t / (0.05 * t + 0.95 * hi), str(round(100 * 0.05 * t / (0.05 * t + 0.95 * hi))), "PPV at 5% prevalence with FPR at the upper CI bound")
R = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
if R:
    c = R["analysis"]["configs"]["det_base"]["fpr@dev_ci_thr"]
    put("x/falseflags/base/dev/lo", c[0], str(round(1000 * c[0])), "results.json det_base fpr@dev_ci_thr[0] x 1000")
    put("x/falseflags/base/dev/hi", c[1], str(round(1000 * c[1])), "results.json det_base fpr@dev_ci_thr[1] x 1000")
json.dump(V, open(f"{OUT}/results_extra.json", "w"), indent=1, ensure_ascii=False)
with open(f"{OUT}/results_extra.tex", "w") as f:
    f.write("\\makeatletter\n")
    for k in sorted(V):
        f.write(f"\\@namedef{{res@{k}}}{{{V[k]['s']}}}\n")
    f.write("\\makeatother\n")
print(len(V), "keys")

if len(sys.argv) > 6:
    AD = sys.argv[6]
    from collections import Counter
    a1 = [json.loads(l) for l in open(f"{AD}/audit1_human.jsonl")]
    put("x/aud1/n-h", sum(x["kind"] == "human" for x in a1), str(sum(x["kind"] == "human" for x in a1)), "audit1_human.jsonl: human items")
    put("x/aud1/n-m", sum(x["kind"] == "machine" for x in a1), str(sum(x["kind"] == "machine" for x in a1)), "audit1_human.jsonl: machine items")
    ng = len({x["generator"] for x in a1 if x["kind"] == "machine"})
    put("x/aud1/n-gens", ng, {4: "four", 5: "five", 6: "six"}.get(ng, str(ng)), "audit1_human.jsonl: distinct generators of machine items", kind="name")
    put("x/aud1/mt-m", sum(x["kind"] == "machine" and x["reason"] == "looks_mt" for x in a1), str(sum(x["kind"] == "machine" and x["reason"] == "looks_mt" for x in a1)), "audit1_human.jsonl: machine items judged to read like machine translation")
    hj = {x["id"]: x for x in a1}
    sc = [json.loads(l) for l in open(f"{AD}/audit1_llm_screen.jsonl")]
    only = [x for x in sc if not x["valid"] and not (x["id"] in hj and not hj[x["id"]]["valid"])]
    put("x/aud1/screen-only-h", sum(x["kind"] == "human" for x in only), str(sum(x["kind"] == "human" for x in only)), "audit1_llm_screen.jsonl: human items removed on the screen alone")
    put("x/aud1/screen-only-m", sum(x["kind"] == "machine" for x in only), str(sum(x["kind"] == "machine" for x in only)), "audit1_llm_screen.jsonl: machine items removed on the screen alone")
    a2 = [json.loads(l) for l in open(f"{AD}/audit2_postfilter.jsonl")]
    es = [x for x in a2 if x["kind"] == "human" and x["stratum"] == "essay"]
    put("x/aud2/essay-valid", sum(x["valid"] for x in es), str(sum(x["valid"] for x in es)), "audit2_postfilter.jsonl: valid educational passages")
    put("x/aud2/essay-n", len(es), str(len(es)), "audit2_postfilter.jsonl: educational passages judged")
    json.dump(V, open(f"{OUT}/results_extra.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{OUT}/results_extra.tex", "w") as f:
        f.write("\\makeatletter\n")
        for k in sorted(V):
            f.write(f"\\@namedef{{res@{k}}}{{{V[k]['s']}}}\n")
        f.write("\\makeatother\n")
    print(len(V), "keys after audit")

fams = [("gemma4-e2b", "Gemma-4-E2B"), ("llama3.2-3b", "Llama-3.2-3B"), ("mgpt-1.3b", "mGPT-1.3B"), ("qwen3-4b", "Qwen3-4B")]
stats = [("binoculars", "Binoculars"), ("obs_ll", "log-likelihood"), ("obs_logrank", "log-rank"), ("obs_entropy", "entropy"), ("obs_lrr", "likelihood/log-rank"),
         ("obs_fdg", "Fast-DetectGPT"), ("perf_ll", "log-likelihood"), ("perf_logrank", "log-rank"), ("perf_entropy", "entropy"), ("perf_lrr", "likelihood/log-rank"),
         ("perf_fdg", "Fast-DetectGPT")]
L = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{llcccc}\toprule",
     r" & Statistic & " + " & ".join(n for _, n in fams) + r"\\\midrule"]
for s, name in stats:
    if s == "obs_ll": L.append(r"\multicolumn{6}{l}{\textit{observer}}\\")
    if s == "perf_ll": L.append(r"\multicolumn{6}{l}{\textit{performer}}\\")
    L.append(f" & {name} & " + " & ".join(r"\res{x/zs/%s:%s/tpr} (\res{x/zs/%s:%s/auroc})" % (f, s, f, s) for f, _ in fams) + r"\\")
L += [r"\bottomrule\end{tabular}",
      r"\caption{All \res{x/zs/n-stats} zero-shot statistics: TPR (\%) on in-distribution test machine text at a threshold giving \res{const/target-fpr}\% FPR on training-split humans, and AUROC in parentheses. Each statistic's direction is fixed on training-split data. Test FPRs range from \res{x/zs/min-fpr}\% to \res{x/zs/max-fpr}\%. For mGPT the performer is \texttt{mGPT-1.3B-kazakh}.}\label{tab:zsall}\end{table*}"]
open(f"{OUT}/table_zs.tex", "w").write("\n".join(L) + "\n")
rows = [("mDeBERTa, calibrated", "x/ci/base"), ("Gemma-4-E2B Binoculars", "x/ci/bino"), ("+ script aug., length matching and thresholds", "x/dep/auglen"), ("AINA-KK-Det", "x/dep/combo")]
L = [r"\begin{table*}[t]\centering\small\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{\begin{tabular}{l|ccc|ccc|ccc}\toprule",
     r" & \multicolumn{3}{c|}{KazLLM} & \multicolumn{3}{c|}{Sherkala} & \multicolumn{3}{c}{Kazakh-built, pooled}\\",
     r"Detector & caught & exact CI & re-fit CI & caught & exact CI & re-fit CI & caught & exact CI & re-fit CI\\\midrule"]
for name, p in rows:
    cells = []
    for g in ("kazllm", "sherkala", "kz"):
        cells += [r"\res{%s/%s/k}/\res{%s/%s/n}" % (p, g, p, g), r"[\res{%s/%s/cp/lo}, \res{%s/%s/cp/hi}]" % (p, g, p, g),
                  r"[\res{%s/%s/boot/lo}, \res{%s/%s/boot/hi}]" % (p, g, p, g) if p != "x/ci/bino" else "--"]
    L.append(name + " & " + " & ".join(cells) + r"\\")
L += [r"\bottomrule\end{tabular}}",
      r"\caption{Detection of the Kazakh-built test outputs with 95\% intervals: exact (Clopper--Pearson, passages only) and a bootstrap over source documents that also resamples the development humans and re-fits the thresholds (as for FPRs). Rows 3--4 use length-specific thresholds on every input. The re-fit intervals of the single classifiers are wide because their thresholds sit in a band of saturated probabilities. Binoculars has no re-fit interval because its threshold is fitted on training-split humans, which this bootstrap does not resample.}\label{tab:ci}\end{table*}"]
open(f"{OUT}/table_ci.tex", "w").write("\n".join(L) + "\n")
print("wrote table_zs.tex and table_ci.tex")
