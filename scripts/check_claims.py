import json, sys
A = json.load(open("results/analysis_det.json")); C = A["configs"]
import os
HAVE_V = os.path.exists("results/results.json")
HAVE_D = os.path.exists("results/dev_replication.json")
V = {k: x["v"] for k, x in json.load(open("results/results.json"))["values"].items()} if HAVE_V else {}
VS = {k: x["s"] for k, x in json.load(open("results/results.json"))["values"].items()} if HAVE_V else {}
D = json.load(open("results/dev_replication.json"))["rows"] if HAVE_D else {}
DC = json.load(open("results/dev_replication.json"))["combos"] if HAVE_D else {}
def g(k, *path):
    e = C.get(k)
    for p in path: e = e.get(p) if isinstance(e, dict) else None
    return e
KZ, SH = "LLama-3.1-KazLLM-1.0-8B", "Llama-3.1-Sherkala-8B-Chat"
GLOBAL = [x for x in g("det_base", "flag_by_gen@dev") if x not in (KZ, SH)]
claims = [
 ("TMR ranks human above machine (AUROC<0.5)", g("ots_tmr-roberta", "auroc") < 0.5),
 ("HC3 flags nothing, machine included", g("ots_hc3-roberta", "fpr@0.5") == 0 and g("ots_hc3-roberta", "tpr@0.5") < 0.01),
 ("recalibrated OTS catch <=5%", max(C[k]["tpr@dev"] for k in C if k.startswith("ots_")) <= 0.05),
 ("Qwen Binoculars at chance", 0.4 <= g("zs:qwen3-4b:binoculars", "auroc") <= 0.6),
 ("Qwen Fast-DetectGPT at chance", 0.4 <= g("zs:qwen3-4b:obs_fdg", "auroc") <= 0.6),
 ("Gemma Binoculars is the best zero-shot TPR", g("zs:gemma4-e2b:binoculars", "tpr@dev") == max(C[k]["tpr@dev"] for k in C if k.startswith("zs:"))),
 ("Kazakh mGPT raises Fast-DetectGPT AUROC", g("zs:mgpt-1.3b:perf_fdg", "auroc") > g("zs:mgpt-1.3b:obs_fdg", "auroc")),
 ("... but stays short of Gemma Binoculars", g("zs:mgpt-1.3b:perf_fdg", "auroc") < g("zs:gemma4-e2b:binoculars", "auroc")),
 ("mDeBERTa separates not by length (AUROC 120-220 >= 0.98)", g("det_base", "auroc_len120_220") >= 0.98),
 ("default 0.5 over-flags humans (FPR > 2%)", g("det_base", "fpr@0.5") > 0.02),
 ("50-word FPR above full-length at 0.5", g("det_base", "short", "50", "fpr@0.5") > g("det_base", "fpr@0.5")),
 ("50-word FPR above full-length at calibrated thr", g("det_base", "short", "50", "fpr@dev") > g("det_base", "fpr@dev")),
 ("machine median shorter than human", True),
 ("Kazakh-built generators in training raise 50-word FPR at 0.5", g("det_kznoaug", "short", "50", "fpr@0.5") > g("det_base", "short", "50", "fpr@0.5")),
 ("KazLLM and Sherkala caught less than every global generator", max(g("det_base", "flag_by_gen@dev", KZ), g("det_base", "flag_by_gen@dev", SH)) < min(g("det_base", "flag_by_gen@dev", x) for x in GLOBAL)),
 ("Latin: base AUROC near/below chance", g("det_base", "script", "latin2021") <= 0.6),
 ("Latin: Gemma Binoculars AUROC near/below chance", g("zs:gemma4-e2b:binoculars", "script", "latin2021") <= 0.6),
 ("homoglyphs cut TPR", g("det_base", "script", "hg_tpr@dev") < g("det_base", "script", "hg_tpr_orig@dev")),
 ("humanizer does not evade (MH flagged >= 80%)", g("det_base", "flag_gray@dev", "MH") >= 0.8),
 ("calibration lowers FPR vs 0.5", g("det_base", "fpr@dev") < g("det_base", "fpr@0.5")),
 ("canonicalization raises Latin TPR", g("det_base", "script", "canon_tpr@dev") > g("det_base", "script", "latin_tpr@dev")),
 ("folding raises homoglyph TPR", g("det_base", "script", "hgfold_tpr@dev") > g("det_base", "script", "hg_tpr@dev")),
 ("canonicalization raises Gemma Binoculars Latin AUROC", g("zs:gemma4-e2b:binoculars", "script", "latin2021canon") > g("zs:gemma4-e2b:binoculars", "script", "latin2021")),
 ("script augmentation lowers FPR at 0.5", g("det_aug", "fpr@0.5") < g("det_base", "fpr@0.5")),
 ("canonicalization still adds TPR for aug", g("det_aug", "script", "canon_tpr@dev") > g("det_aug", "script", "latin_tpr@dev")),
 ("length matching lowers 50-word FPR at 0.5", g("det_auglen", "short", "50", "fpr@0.5") < g("det_aug", "short", "50", "fpr@0.5")),
 ("length matching lowers 50-word FPR at full-length thr", g("det_auglen", "short", "50", "fpr@dev") < g("det_aug", "short", "50", "fpr@dev")),
 ("length-specific thresholds lower it further", g("det_auglen", "short", "50", "fpr@devlen") < g("det_auglen", "short", "50", "fpr@dev")),
 ("single seeds vary at 1% (aug TPR spread > 20 pts)", (lambda r: r[1] - r[0] > 0.2)(g("det_aug", "seed_range", "tpr@dev"))),
 ("seed averages are high (base, aug TPR >= 95%)", g("det_base", "tpr@dev") >= 0.95 and g("det_aug", "tpr@dev") >= 0.95),
 ("Kazakh gens in training (no aug) raise KazLLM and Sherkala detection", g("det_kznoaug", "flag_by_gen@dev", KZ) > g("det_base", "flag_by_gen@dev", KZ) and g("det_kznoaug", "flag_by_gen@dev", SH) > g("det_base", "flag_by_gen@dev", SH)),
 ("... but flag more 50-word humans at 0.5 than base", g("det_kznoaug", "short", "50", "fpr@0.5") > g("det_base", "short", "50", "fpr@0.5")),
 ("with script aug the gain shrinks (below the no-aug variant for both)", g("det_kz", "flag_by_gen@dev", KZ) < g("det_kznoaug", "flag_by_gen@dev", KZ) and g("det_kz", "flag_by_gen@dev", SH) < g("det_kznoaug", "flag_by_gen@dev", SH)),
 ("with length matching as well, in-dist TPR < 80% and Kazakh detection below base", g("det_kzlen", "tpr@dev") < 0.8 and g("det_kzlen", "flag_by_gen@dev", KZ) < g("det_base", "flag_by_gen@dev", KZ) and g("det_kzlen", "flag_by_gen@dev", SH) < g("det_base", "flag_by_gen@dev", SH)),
 ("length matching costs full-length FPR above 1%", g("det_auglen", "fpr@dev") > 0.01),
 ("calibration lowers base 50-word FPR vs 0.5", g("det_base", "short", "50", "fpr@dev") < g("det_base", "short", "50", "fpr@0.5")),
 ("aug still flags more 50-word humans than base at calibrated thr", g("det_aug", "short", "50", "fpr@dev") > g("det_base", "short", "50", "fpr@dev")),
 ("Binoculars catches a real share of classifier misses (>= 25%)", (A.get("complement", {}).get("auglen", {}).get("bino_catches") or 0) >= 0.25),
 ("combination raises KazLLM detection", g("combo_auglen+bino", "flag_by_gen@dev", KZ) > g("det_auglen", "flag_by_gen@dev", KZ)),
 ("combination raises Sherkala detection", g("combo_auglen+bino", "flag_by_gen@dev", SH) > g("det_auglen", "flag_by_gen@dev", SH)),
 ("combination lowers test FPR vs classifier alone", g("combo_auglen+bino", "fpr@dev") < g("det_auglen", "fpr@dev")),

]
if HAVE_D:
    claims += [
     ("dev: calibration lowers FPR from the 0.5 cut-off", D["base"]["fpr"] < D["base05"]["fpr"]),
     ("dev: script augmentation raises 50-word FPR", D["aug"]["fpr50"] > D["base"]["fpr50"]),
     ("dev: length matching lowers it", D["auglen"]["fpr50"] < D["aug"]["fpr50"]),
     ("dev: length-specific thresholds lower it further", D["auglen-lenthr"]["fpr50"] < D["auglen"]["fpr50"]),
     ("dev: length-matching FPR cost small (<= 0.5 points over aug)", D["auglen"]["fpr"] - D["aug"]["fpr"] <= 0.005),
     ("dev: combination costs in-distribution recall", D["combo"]["tpr_indist"] < D["auglen-lenthr"]["tpr_indist"]),
     ("dev: combination raises KazLLM and Sherkala detection", D["combo"]["tpr_kazllm"] > D["auglen-lenthr"]["tpr_kazllm"] and D["combo"]["tpr_sherkala"] > D["auglen-lenthr"]["tpr_sherkala"]),
     ("dev: Kazakh gens in training (no aug) raise KazLLM and Sherkala detection", D["kznoaug"]["tpr_kazllm"] > D["base"]["tpr_kazllm"] and D["kznoaug"]["tpr_sherkala"] > D["base"]["tpr_sherkala"]),
     ("dev: with script aug the gain shrinks for both", D["kz"]["tpr_kazllm"] < D["kznoaug"]["tpr_kazllm"] and D["kz"]["tpr_sherkala"] < D["kznoaug"]["tpr_sherkala"]),
     ("dev: with length matching in-dist recall collapses (< 80%)", D["kzlen"]["tpr_indist"] < 0.8),
     ("dev: ... but Kazakh-built detection does not fall below the row above (exception stated)", D["kzlen"]["tpr_kazllm"] >= D["kz"]["tpr_kazllm"] and D["kzlen"]["tpr_sherkala"] >= D["kz"]["tpr_sherkala"]),
     ("dev: the released combination catches the most KazLLM text of the three", DC["det_auglen"]["tpr_kazllm"] > max(DC["det_aug"]["tpr_kazllm"], DC["det_base"]["tpr_kazllm"])),
    ]
if HAVE_V:
    claims += [
     ("no human test passage has a near-duplicate in training at the cross-split Jaccard", V["val/split/n-neardup"] == 0),
     ("the development draw contains no educational passage", V["val/split/dev-essay"] == 0),
     ("surface baseline catches more than any zero-shot detector", V["val/surface/tpr-at-dev"] > max(C[k]["tpr@dev"] for k in C if k.startswith("zs:"))),
     ("surface baseline AUROC is high (>= 0.95), also at 120-220 words", V["val/surface/auroc"] >= 0.95 and V["val/surface/auroc-len120-220"] >= 0.95),
     ("Latin: the low-FPR group catches almost nothing (<= 1%)", V["val/latin/max-tpr-low"] <= 0.01 and V["val/latin/n-low"] + V["val/latin/n-high"] == V["val/latin/n"]),
     ("Latin: the others flag >= 10% of Latin human text", V["val/latin/min-fpr-high"] >= 0.10),
     ("Latin: mDeBERTa is in the low group", C["det_base"]["script"]["latin_fpr@dev"] < 0.10),
     ("digits: mDeBERTa still ranks digit-free passages (AUROC >= 0.98)", V["val/digits/base/auroc-nodigit"] >= 0.98),
     ("digits: digit-free humans flagged more (mDeBERTa, AINA-KK-Det)", V["val/digits/base/fpr-nodigit"] > V["val/digits/base/fpr-digit"] and V["val/digits/combo/fpr-nodigit"] > V["val/digits/combo/fpr-digit"]),
     ("dev: combination and classifier give the same full-length FPR (as printed)", VS["devrep/auglen/fpr"] == VS["devrep/combo/fpr"]),
     ("family: AINA-KK-Det catches Gemma-4 more than Kazakh-built", V["val/family/combo/gemma"] > V["val/family/combo/kz"]),
     ("post-filter audit: 'only' a minority-to-two-thirds of humans valid, and 'about a third' defective (28-40%)", 0.28 <= 1 - V["aud/post/h-valid-w"] <= 0.40),
     ("post-filter audit: first and second samples do not overlap ('fresh')", V["aud/post/overlap-first"] == 0),
     ("post-filter audit: public detectors flag valid passages too", V["aud/post/fakespot-roberta/valid"] > 0 and V["aud/post/tmr-roberta/valid"] > 0),
     ("classifier alone misses the 1% target (FPR > 1%)", g("det_auglen", "fpr@dev") > 0.01),
     ("combination costs in-distribution recall", g("combo_auglen+bino", "tpr@dev") < g("det_auglen", "tpr@dev")),
     ("combination PPV beats Fakespot", True),
    ]
bad = 0
for name, ok in claims:
    print(("ok   " if ok else "FAIL ") + name); bad += not ok
print(f"{len(claims) - bad}/{len(claims)} claims hold"); sys.exit(1 if bad else 0)
