import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))
V0 = R["values"]
X = json.load(open("paper/results_extra.json")) if os.path.exists("paper/results_extra.json") else {}
V = {**V0, **X}
GK = R["gen_keys"]
CELLS = []

def key(cfg, *path):
    pre = f"ots/{cfg[4:]}" if cfg.startswith("ots_") else f"det/{cfg.replace('det_', '').replace('_', '-')}"
    return "/".join([pre] + [str(GK.get(p, p)).replace("_", "-").replace("@", "-at-") for p in path])

def cell(table, row, col, k):
    if k is None:
        CELLS.append({"table": table, "row": row, "col": col, "key": None, "value": "--"})
        return "--"
    s = V[k]["s"]
    CELLS.append({"table": table, "row": row, "col": col, "key": k, "value": s})
    return s

def cap(table, k):
    CELLS.append({"table": table, "row": "caption", "col": "", "key": k, "value": V[k]["s"]})
    return V[k]["s"]

with open("paper/results.tex", "w", encoding="utf-8") as f:
    f.write("\\makeatletter\n\\newcommand{\\res}[1]{\\@ifundefined{res@#1}{\\textbf{??#1??}}{\\@nameuse{res@#1}}}\n")
    for k in sorted(V0):
        f.write(f"\\@namedef{{res@{k}}}{{{V0[k]['s']}}}\n")
    f.write("\\makeatother\n")
    names = [R["ots_names"][k][0] for k in R["ots_names"]]
    ids = [f"\\url{{{R['ots_names'][k][1]}}}" for k in R["ots_names"]]
    join = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1]
    f.write(f"\\newcommand{{\\RESotsList}}{{{join(names)}}}\n\\newcommand{{\\RESotsIDs}}{{{join(ids)}}}\n")
    for alias, k in (("RESnHuman", "corpus/n-human"), ("RESnMachine", "corpus/n-nonhuman"), ("RESnGen", "corpus/n-gen"),
                     ("RESnOutlets", "corpus/n-outlets"), ("RESnEduHosts", "corpus/n-edu-hosts")):
        f.write(f"\\newcommand{{\\{alias}}}{{\\res{{{k}}}}}\n")

T = "tab:gens"
L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.2pt}", "\\begin{tabular}{llrrrrr}\\toprule",
     "Generator & Access & M & MIX & HP & MT & MH\\\\\\midrule"]
for row in R["table1"]:
    mark = "$^\\dagger$" if row["mark"] == "held" else ("$^\\ddagger$" if row["mark"] == "gray" else "")
    L.append(f"{row['name']}{mark} & {row['access']} & " + " & ".join(cell(T, row["name"], c, row[c]) for c in ("M", "MIX", "HP", "MT", "MH")) + "\\\\")
L.append(f"\\midrule Human & -- & \\multicolumn{{5}}{{l}}{{{cell(T, 'Human', 'passages', 'corpus/n-human')} passages ({cell(T, 'Human', 'domains', 'corpus/n-domains-n')} domains)}}\\\\")
L += ["\\bottomrule\\end{tabular}",
      "\\caption{Generators and passage counts after QC. M: machine text (C1, C2); MIX: human opening + machine continuation; HP: AI-polished human text; "
      "MT: machine translation from Russian (C4 LLM, C5 NLLB); MH: humanizer rewrite of M. Access: Gemini or Groq API, or open weights. "
      "$^\\dagger$Held out: never used for training except in the last three rows of Table~\\ref{tab:fixes}. Model cards: \\citet{gemini31fl2026,gemini35fl2026,gemma4card2026,openai2025gptoss,qwen38card2026}. $^\\ddagger$Gray-zone text only.}", "\\label{tab:gens}\\end{table}"]
open("paper/table_generators.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

T = "tab:main"
ZS = [("zs:gemma4-e2b:binoculars", "Gemma-4-E2B / Binoculars"), ("zs:gemma4-e2b:obs_fdg", "Gemma-4-E2B / Fast-DetectGPT"),
      ("zs:llama3.2-3b:binoculars", "Llama-3.2-3B / Binoculars"), ("zs:llama3.2-3b:obs_fdg", "Llama-3.2-3B / Fast-DetectGPT"),
      ("zs:qwen3-4b:binoculars", "Qwen3-4B / Binoculars"), ("zs:qwen3-4b:obs_fdg", "Qwen3-4B / Fast-DetectGPT"),
      ("zs:mgpt-1.3b:obs_fdg", "mGPT-1.3B / Fast-DetectGPT"), ("zs:mgpt-1.3b:perf_fdg", "mGPT-1.3B-kazakh / Fast-DetectGPT")]
FT = [("det_xlmr", "XLM-R-base (fine-tuned)"), ("det_base", "mDeBERTa-v3-base (fine-tuned)")]
OTS = [("ots_fakespot-roberta", "Fakespot RoBERTa"), ("ots_openai-roberta", "OpenAI RoBERTa (GPT-2 output)"),
       ("ots_hc3-roberta", "HC3 RoBERTa (ChatGPT)"), ("ots_tmr-roberta", "TMR RoBERTa"), ("ots_desklib-deberta", "Desklib DeBERTa")]
DOMS = (("encyclopedic", "enc."), ("news", "news"), ("web", "web"), ("essay", "edu."))

def main_row(k, nm, op):
    o = "dev" if op == "cal" else "0.5"
    c = lambda col, *p: cell(T, nm, col, key(k, *p))
    return (f"{nm} & {c('AUROC', 'auroc')} [{c('AUROC CI lo', 'auroc_ci', 'lo')}, {c('AUROC CI hi', 'auroc_ci', 'hi')}] & {c('AUROC held-out', 'auroc_heldout')} & {op} & "
            f"{c('FPR all', f'fpr@{o}')} & " + " & ".join(c(f"FPR {lab}", f"fpr_by_domain@{o}", d) for d, lab in DOMS) +
            f" & {c('FPR 50 words', 'short', '50', f'fpr@{o}')} & {c('TPR in-dist.', f'tpr@{o}')} & {c('TPR held-out', f'tpr_heldout@{o}')}\\\\")

L = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccc|ccccc|c|cc}\\toprule",
     " & \\multicolumn{2}{c}{AUROC} & & \\multicolumn{6}{c|}{FPR on human test passages (\\%)} & \\multicolumn{2}{c}{TPR (\\%)}\\\\",
     "\\cmidrule(lr){2-3}\\cmidrule(lr){5-10}\\cmidrule(lr){11-12}",
     "Detector & in-dist.\\ [95\\% CI] & held-out & thr. & all & enc. & news & web & edu. & 50 words & in-dist. & held-out\\\\\\midrule",
     f"\\multicolumn{{12}}{{l}}{{\\textit{{Zero-shot, threshold at {V['const/target-fpr']['s']}\\% FPR on training-split human text}}}}\\\\"]
L += [main_row(k, nm, "cal") for k, nm in ZS]
L += ["\\midrule", "\\multicolumn{12}{l}{\\textit{Fine-tuned on AINA-KK train, default threshold}}\\\\"]
L += [main_row(k, nm, "0.5") for k, nm in FT]
L += ["\\midrule", "\\multicolumn{12}{l}{\\textit{Off-the-shelf public detectors, as released}}\\\\"]
L += [main_row(k, nm, "0.5") for k, nm in OTS]
L += ["\\bottomrule\\end{tabular}}",
      f"\\caption{{Detectors as an institution would deploy them. Human: {cap(T, 'det/n/H-test')} test-split passages (50 words: {cap(T, 'corpus/variant/H/trunc50')} of them cut to their first "
      f"{cap(T, 'const/trunc-short')} words). In-distribution machine text: {cap(T, 'det/n/M-indist')} passages from the {cap(T, 'corpus/n-train-gens')} training generators; "
      f"held-out: {cap(T, 'det/n/M-heldout-topicdisjoint')} passages from the {cap(T, 'corpus/n-heldout-gens')} unseen generators on test-split sources. "
      "AUROC CIs: bootstrap over source documents.}\\label{tab:main}\\end{table*}"]
open("paper/table_main.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

T = "tab:stress"
STRESS = [("zs:gemma4-e2b:binoculars", "Gemma-4-E2B / Binoculars"), ("zs:mgpt-1.3b:perf_fdg", "mGPT-1.3B-kazakh / Fast-DetectGPT"),
          ("det_xlmr", "XLM-R-base (FT)"), ("det_base", "mDeBERTa-v3-base (FT)")]
L = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\resizebox{\\textwidth}{!}{\\begin{tabular}{l|cccc|cc|cc|cc}\\toprule",
     " & \\multicolumn{4}{c|}{Gray zones: flagged (\\%)} & \\multicolumn{2}{c|}{Kazakh-built: TPR} & \\multicolumn{2}{c|}{Latin-2021} & \\multicolumn{2}{c}{Homoglyphs: TPR} \\\\",
     "Detector & HP & MT & MIX & MH & KazLLM & Sherkala & AUROC & TPR & before & after\\\\\\midrule"]
for k, nm in STRESS:
    c = lambda col, *p: cell(T, nm, col, key(k, *p))
    L.append(f"{nm} & " + " & ".join(c(g, "flag_gray@dev", g) for g in ("HP", "MT", "MIX", "MH")) +
             f" & {c('KazLLM', 'flag_by_gen@dev', 'LLama-3.1-KazLLM-1.0-8B')} & {c('Sherkala', 'flag_by_gen@dev', 'Llama-3.1-Sherkala-8B-Chat')}"
             f" & {c('Latin AUROC', 'script', 'latin2021')} & {c('Latin TPR', 'script', 'latin_tpr@dev')} & {c('HG before', 'script', 'hg_tpr_orig@dev')} & {c('HG after', 'script', 'hg_tpr@dev')}\\\\")
L += ["\\bottomrule\\end{tabular}}",
      f"\\caption{{Stress tests at the calibrated threshold ({cap(T, 'const/target-fpr')}\\% FPR on unseen human text). HP: AI-polished human text; MT: machine-translated from Russian; "
      f"MIX: human opening ($\\sim${cap(T, 'const/c2-prefix')} words) continued by a model; MH: machine text rewritten by a second model to sound human. "
      f"Latin-2021: both classes transliterated ({cap(T, 'corpus/variant/H/latin2021')} human + {cap(T, 'corpus/variant/M/latin2021')} machine passages). "
      f"Homoglyphs: {cap(T, 'corpus/variant/M/homoglyph10')} machine passages before and after {cap(T, 'const/hg-attack')}\\% of eligible Cyrillic letters are replaced by Latin look-alikes.}}"
      "\\label{tab:stress}\\end{table*}"]
open("paper/table_stress.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

T = "tab:fixes"
FIX = [("det_base", "mDeBERTa, threshold 0.5", "0.5", False, False),
       ("det_base", "\\quad + threshold calibrated on unseen humans", "dev", False, False),
       ("det_base", "\\quad + script canonicalization", "dev", True, False),
       ("det_aug", "\\quad + script-augmented training", "dev", True, False),
       ("det_auglen", "\\quad + length-matched training", "dev", True, False),
       ("det_auglen", "\\quad + length-specific thresholds", "dev", True, True),
       ("combo_auglen+bino", "\\quad + OR Gemma-4-E2B Binoculars (\\textbf{AINA-KK-Det})", "dev", True, True),
       ("det_kznoaug", "mDeBERTa + KazLLM and Sherkala in training", "dev", True, False),
       ("det_kz", "\\quad + script-augmented training", "dev", True, False),
       ("det_kzlen", "\\quad + length-matched, length-specific thresholds", "dev", True, True)]
L = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\resizebox{\\textwidth}{!}{\\begin{tabular}{l|ccc|ccc|cc|cc}\\toprule",
     " & \\multicolumn{3}{c|}{FPR on human text (\\%)} & \\multicolumn{3}{c|}{TPR (\\%)} & \\multicolumn{2}{c|}{Kazakh-built: TPR} & \\multicolumn{2}{c}{Script: TPR} \\\\",
     "Configuration & all [95\\% CI] & web & 50 words & in-dist. & held-out$^\\ast$ & 50 words & KazLLM & Sherkala & Latin & homoglyph\\\\\\midrule"]
DEP = {"det_auglen": "x/dep/auglen", "combo_auglen+bino": "x/dep/combo", "det_kzlen": "x/dep/kzlen"}
for k, nm, op, canon, lenthr in FIX:
    rl = nm.replace("\\quad + ", "+ ")
    if lenthr and X and k in DEP:
        d = DEP[k]; cc = lambda col, s: cell(T, rl, col, f"{d}/{s}")
        L.append(f"{nm} & {cc('FPR all', 'fpr')} [{cc('FPR CI lo', 'fpr-ci/lo')}, {cc('FPR CI hi', 'fpr-ci/hi')}] & {cc('FPR web', 'fpr-web')} & {cc('FPR 50 words', 'fpr50')} & "
                 f"{cc('TPR in-dist.', 'tpr')} & {cc('TPR held-out', 'tpr-heldout')} & {cc('TPR 50 words', 'tpr50')} & {cc('KazLLM', 'kazllm')} & {cc('Sherkala', 'sherkala')} & "
                 f"{cc('Latin', 'latin-canon')} & {cc('homoglyph', 'hgfold')}\\\\")
        if k.startswith("combo_"):
            L.append("\\midrule")
        continue
    c = lambda col, *p: cell(T, rl, col, key(k, *p))
    ci = f" [{c('FPR CI lo', f'fpr@{op}_ci_thr', 'lo')}, {c('FPR CI hi', f'fpr@{op}_ci_thr', 'hi')}]" if op == "dev" else ""
    s50 = "fpr@devlen" if lenthr else f"fpr@{op}"
    t50 = "tpr@devlen" if lenthr else f"tpr@{op}"
    if op == "0.5":
        lat, hg = cell(T, rl, "Latin", None), cell(T, rl, "homoglyph", None)
    else:
        lat = c("Latin", "script", "canon_tpr@dev" if canon else "latin_tpr@dev")
        hg = c("homoglyph", "script", "hgfold_tpr@dev" if canon else "hg_tpr@dev")
    L.append(f"{nm} & {c('FPR all', f'fpr@{op}')}{ci} & {c('FPR web', f'fpr_by_domain@{op}', 'web')} & {c('FPR 50 words', 'short', '50', s50)} & "
             f"{c('TPR in-dist.', f'tpr@{op}')} & {c('TPR held-out', f'tpr_heldout@{op}')} & {c('TPR 50 words', 'short', '50', t50)} & "
             f"{c('KazLLM', f'flag_by_gen@{op}', 'LLama-3.1-KazLLM-1.0-8B')} & {c('Sherkala', f'flag_by_gen@{op}', 'Llama-3.1-Sherkala-8B-Chat')} & {lat} & {hg}\\\\")
    if k.startswith("combo_"):
        L.append("\\midrule")
L += ["\\bottomrule\\end{tabular}}",
      f"\\caption{{Fixes, applied cumulatively to mDeBERTa-v3-base ({cap(T, 'const/seeds-digit')} seeds, probabilities averaged). Calibrated thresholds give {cap(T, 'const/target-fpr')}\\% FPR on development "
      f"humans ({cap(T, 'det/n/H-dev')} passages from training topics held out of training); length-specific thresholds are fitted on the same passages cut to 100 or 50 words and applied to every input by its length (\\S\\ref{{sec:protocol}}). "
      "Script canonicalization maps Latin-2021 text back to Cyrillic and folds Latin homoglyphs before scoring. OR: a text is flagged if either "
      f"detector exceeds its own threshold, the two set jointly to {cap(T, 'const/target-fpr')}\\% FPR on development humans. $^\\ast$Held-out: test-source outputs of the four held-out generators; "
      f"in the last three rows KazLLM and Sherkala outputs for training-split sources are also used for training, so only the {cap(T, 'corpus/n-heldout-gemma')} Gemma-4 models remain unseen there. "
      "FPR CI: bootstrap over source documents that also resamples the development humans and re-fits the thresholds, so it includes their sampling error. "
      "Appendix~\\ref{app:devrep} repeats this table on development data alone.}\\label{tab:fixes}\\end{table*}"]
open("paper/table_fixes.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

T = "tab:devrep"
DRR = [("base05", "mDeBERTa, threshold 0.5"), ("base", "\\quad + threshold calibrated on unseen humans"), ("aug", "\\quad + script-augmented training"),
       ("auglen", "\\quad + length-matched training"), ("auglen-lenthr", "\\quad + length-specific thresholds"),
       ("combo", "\\quad + OR Gemma-4-E2B Binoculars (\\textbf{AINA-KK-Det})"), ("kznoaug", "mDeBERTa + KazLLM and Sherkala in training"),
       ("kz", "\\quad + script-augmented training"), ("kzlen", "\\quad + length-matched, length-specific thresholds")]
COLS = [("FPR all", "fpr"), ("FPR web", "fpr-web"), ("FPR 50 words", "fpr50"), ("TPR in-dist.", "tpr-indist"), ("TPR Gemma-4", "tpr-gemma"),
        ("TPR KazLLM", "tpr-kazllm"), ("TPR Sherkala", "tpr-sherkala")]
L = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{5pt}",
     "\\resizebox{\\textwidth}{!}{\\begin{tabular}{l|ccc|cccc}\\toprule",
     " & \\multicolumn{3}{c|}{FPR on development humans (\\%)} & \\multicolumn{4}{c}{TPR on development machine text (\\%)}\\\\",
     "Configuration & all & web & 50 words & in-dist. & Gemma-4 & KazLLM & Sherkala\\\\\\midrule"]
DDEP = {"auglen-lenthr": "x/devdep/auglen", "combo": "x/devdep/combo", "kzlen": "x/devdep/kzlen"}
for r_, nm in DRR:
    rl = nm.replace("\\quad + ", "+ ")
    base = DDEP[r_] if (X and r_ in DDEP) else f"devrep/{r_}"
    L.append(nm + " & " + " & ".join(cell(T, rl, cn, f"{base}/{m_}") for cn, m_ in COLS) + "\\\\")
    if r_ == "combo":
        L.append("\\midrule")
L += ["\\bottomrule\\end{tabular}}",
      f"\\caption{{Table~\\ref{{tab:fixes}} repeated on development data alone: {cap(T, 'devrep/n-h')} human passages from the {cap(T, 'det/n/dev-clusters')} development topic clusters and the "
      f"{cap(T, 'devrep/n-m')} machine passages prompted from them, none of which any configuration was trained on (the last three rows included). Each passage is judged with thresholds, "
      f"and for the OR-combination the empirical distributions, fitted on the other {cap(T, 'devrep/n-fit-clusters')} clusters; length-specific thresholds apply to every input by its length. Development data have no script variants, so the canonicalization row is omitted. "
      f"Machine passages per column: {cap(T, 'devrep/n/tpr-indist')}, {cap(T, 'devrep/n/tpr-gemma')}, {cap(T, 'devrep/n/tpr-kazllm')} and {cap(T, 'devrep/n/tpr-sherkala')}.}}\\label{{tab:devrep}}\\end{{table*}}"]
open("paper/table_devrep.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

T = "tab:cues"
NICE = {"nb_hyphen_U+2011": "non-breaking hyphen (U+2011)", "narrow_nbsp_U+202F": "narrow no-break space", "soft_hyphen_U+00AD": "soft hyphen",
        "ellipsis_char": "ellipsis character", "title_line": "standalone title line", "markdown_header": "markdown header",
        "bold_markup": "bold markup", "trailing_fragment": "trailing fragment", "mixed_script_token": "mixed-script word", "latin_homoglyph": "Latin homoglyph"}
L = ["\\begin{table*}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{6pt}", "\\begin{tabular}{lrrl}\\toprule", "Cue & Human & Machine & Highest generator\\\\\\midrule"]
for c_, v in R["cues"].items():
    if c_ not in NICE:
        continue
    bg = max(v["by_generator"].items(), key=lambda x: x[1])[0]
    L.append(f"{NICE[c_]} & {cell(T, NICE[c_], 'Human', f'cue/{c_}/h')} & {cell(T, NICE[c_], 'Machine', f'cue/{c_}/m')} & "
             f"{R['gen_names'].get(bg, bg)} ({cell(T, NICE[c_], 'Highest generator', f'cue/{c_}/{GK.get(bg, bg)}')})\\\\")
L += ["\\bottomrule\\end{tabular}",
      f"\\caption{{Share of passages (\\%) containing each surface cue before normalization: {cap(T, 'cue/n-human')} encyclopedic, news and web passages before the second cleaning pass vs.\\ "
      f"{cap(T, 'cue/n-machine')} raw C1/C2 outputs.}}\\label{{tab:cues}}\\end{{table*}}"]
open("paper/table_cues.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

T = "tab:logo"
L = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{lcccc}\\toprule",
     " & \\multicolumn{2}{c}{AUROC} & \\multicolumn{2}{c}{TPR (\\%)}\\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Generator & out & in & out & in\\\\\\midrule"]
for g in sorted(R["analysis"]["logo"], key=lambda g: R["gen_names"].get(g, g)):
    nm = R["gen_names"].get(g, g)
    b = f"det/logo/{GK.get(g, g)}"
    L.append(f"{nm} & {cell(T, nm, 'AUROC out', b + '/auroc')} & {cell(T, nm, 'AUROC in', b + '/auroc-when-trained')} & "
             f"{cell(T, nm, 'TPR out', b + '/tpr-at-dev')} & {cell(T, nm, 'TPR in', b + '/tpr-at-dev-when-trained')}\\\\")
L += ["\\bottomrule\\end{tabular}",
      f"\\caption{{Leave-one-generator-out among the {cap(T, 'corpus/n-train-gens')} training generators. mDeBERTa with script augmentation and length matching, trained on three of them (out) or all four (in), "
      f"scored on that generator's outputs for test-split sources against test humans; TPR at the calibrated {cap(T, 'const/target-fpr')}\\%-FPR threshold. One seed for out, {cap(T, 'const/seeds')} for in; single seeds are unstable at a {cap(T, 'const/target-fpr')}\\% threshold (\\S\\ref{{sec:fixes}}).}}"
      "\\label{tab:logo}\\end{table}"]
open("paper/table_logo.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"]})
ROWS = [("ots_fakespot-roberta", "Fakespot (0.5)", "0.5", None), ("ots_openai-roberta", "OpenAI RoBERTa (0.5)", "0.5", None),
        ("det_base", "mDeBERTa (0.5)", "0.5", None), ("det_kznoaug", "mDeBERTa + Kazakh gens (0.5)", "0.5", None),
        ("det_base", "mDeBERTa (calibrated)", "dev", None), ("zs:gemma4-e2b:binoculars", "Gemma Binoculars (calibrated)", "dev", None),
        ("combo_auglen+bino", "AINA-KK-Det", "dev", "devlen")]
lens = [("full", None), ("100 words", "100"), ("50 words", "50")]
fig, ax = plt.subplots(figsize=(3.3, 3.2))
w = 0.8 / len(ROWS)
cols = plt.cm.tab10.colors
for i, (k, nm, op, sop) in enumerate(ROWS):
    ks = [key(k, f"fpr@{op}")] + [key(k, "short", n, f"fpr@{sop or op}") for _, n in lens[1:]]
    vals = []
    for (ln, _), kk in zip(lens, ks):
        CELLS.append({"table": "fig:length", "row": nm, "col": ln, "key": kk, "value": V[kk]["s"]})
        vals.append(V[kk]["v"])
    xs = [j + (i - (len(ROWS) - 1) / 2) * w for j in range(3)]
    ax.bar(xs, [100 * v for v in vals], width=w, label=nm, color=cols[i % 10], edgecolor="black" if nm == "AINA-KK-Det" else "none", linewidth=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels([l for l, _ in lens], fontsize=8)
ax.set_ylabel("human passages flagged (%)", fontsize=8); ax.tick_params(axis="y", labelsize=7.5)
ax.set_yscale("symlog", linthresh=2); ax.set_yticks([0, 1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
ax.axhline(100 * V["const/target-fpr"]["v"], color="grey", lw=0.5, ls="--")
ax.legend(fontsize=6.6, frameon=False, loc="upper center", bbox_to_anchor=(0.44, -0.09), ncol=2, columnspacing=0.6, handlelength=1.0, handletextpad=0.4)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig("paper/fig_length.pdf")
json.dump(CELLS, open("results/table_cells.json", "w"), indent=0, ensure_ascii=False)
print("rendered results.tex,", len(CELLS), "table/figure cells")
