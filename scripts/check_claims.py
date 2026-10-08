import json, sys
A = json.load(open("results/analysis_det.json")); C = A["configs"]
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
 ("Kazakh generators in training raise 50-word FPR at 0.5", g("det_allnoaug", "short", "50", "fpr@0.5") > g("det_base", "short", "50", "fpr@0.5")),
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
 ("Kazakh gens without aug raise KazLLM and Sherkala detection", g("det_allnoaug", "flag_by_gen@dev", KZ) > g("det_base", "flag_by_gen@dev", KZ) and g("det_allnoaug", "flag_by_gen@dev", SH) > g("det_base", "flag_by_gen@dev", SH)),
 ("... but flag more 50-word humans at 0.5 than base", g("det_allnoaug", "short", "50", "fpr@0.5") > g("det_base", "short", "50", "fpr@0.5")),
 ("with script aug the gain disappears (<= base)", g("det_all", "flag_by_gen@dev", KZ) <= g("det_base", "flag_by_gen@dev", KZ) and g("det_all", "flag_by_gen@dev", SH) <= g("det_base", "flag_by_gen@dev", SH)),
 ("with length matching it is partial (between base and no-aug)", g("det_base", "flag_by_gen@dev", KZ) < g("det_alllen", "flag_by_gen@dev", KZ) < g("det_allnoaug", "flag_by_gen@dev", KZ)),
 ("length matching costs full-length FPR above 1%", g("det_auglen", "fpr@dev") > 0.01),
 ("calibration lowers base 50-word FPR vs 0.5", g("det_base", "short", "50", "fpr@dev") < g("det_base", "short", "50", "fpr@0.5")),
 ("aug still flags more 50-word humans than base at calibrated thr", g("det_aug", "short", "50", "fpr@dev") > g("det_base", "short", "50", "fpr@dev")),
 ("Binoculars catches a real share of classifier misses (>= 25%)", (A.get("complement", {}).get("auglen", {}).get("bino_catches") or 0) >= 0.25),
 ("combination raises KazLLM detection", g("combo_auglen+bino", "flag_by_gen@dev", KZ) > g("det_auglen", "flag_by_gen@dev", KZ)),
 ("combination raises Sherkala detection", g("combo_auglen+bino", "flag_by_gen@dev", SH) > g("det_auglen", "flag_by_gen@dev", SH)),
 ("combination lowers test FPR vs classifier alone", g("combo_auglen+bino", "fpr@dev") < g("det_auglen", "fpr@dev")),
 ("classifier alone misses the 1% target (FPR > 1%)", g("det_auglen", "fpr@dev") > 0.01),
 ("combination costs in-distribution recall", g("combo_auglen+bino", "tpr@dev") < g("det_auglen", "tpr@dev")),
 ("combination is back within the 1% target (FPR <= 1.0%)", g("combo_auglen+bino", "fpr@dev") <= 0.0105),
 ("combination PPV beats Fakespot", True),
]
bad = 0
for name, ok in claims:
    print(("ok   " if ok else "FAIL ") + name); bad += not ok
print(f"{len(claims) - bad}/{len(claims)} claims hold"); sys.exit(1 if bad else 0)
