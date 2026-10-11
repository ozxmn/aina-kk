# AINA-KK

Code for **"Unchecked or Falsely Accused? Evaluating AI-Generated Text Detectors on Kazakh"** (Mansur Ozaman, Nazarbayev University).

AINA-KK is a benchmark for machine-generated text detection in Kazakh: 3,811 human passages written before 30 November 2022 (encyclopedic, news, web and educational text) and 5,597 machine and gray-zone passages from 9 generators, with script (Latin-2021, homoglyph) and length stress tests. This repository contains the pipeline that builds the benchmark, the detector experiments and the analysis.

- Data and detector scores: https://huggingface.co/datasets/ajnasiou/aina-kk
- Detector (AINA-KK-Det): https://huggingface.co/ajnasiou/aina-kk-det

## Setup

```
pip install -r requirements.txt
```

Language identification uses GlotLID (`cis-lmu/glotlid` on the Hugging Face Hub). Generation scripts read API keys from the environment (`GEMINI_API_KEY`, `GROQ_API_KEY`); gated open-weight models need `HF_TOKEN`.

## Pipeline

All commands run from the repository root.

1. **Human text.** `wiki_dump.py` (Kazakh Wikipedia dump of 2022-05-01), `fw2_news_local.py` and `fw2_essay_local.py` (FineWeb2 `kaz_Cyrl` records crawled before 2022-11-30), `cc100_sample.py` (CC100-kk). Then `qc_human.py`, `split_human.py`, `clean_human.py` and `qc_essay.py` filter, deduplicate and split the passages by topic cluster.
2. **Generation.** `build_tasks.py` builds prompts from `prompts.py` (separate train and test templates); `gen.py` calls API models; `kernels/generate_open_models.py` runs KazLLM, Sherkala and NLLB on a GPU; `build_c6.py` and `nllb_tasks.py` build the humanizer and translation tasks; `ru_pool.py` samples the Russian source text.
3. **Benchmark file.** `qc_machine.py` filters the outputs; `build_eval.py` writes `data/eval/eval.jsonl` with the stress-test variants. `textnorm.py` holds the normalization applied to both classes, `translit2021.py` the Latin-2021 transliteration and its inverse.
4. **Detectors.** `kernels/score_zero_shot.py` (with `kernels/zs.py`) computes zero-shot statistics, `kernels/train_finetuned.py` trains the fine-tuned detectors (set `PART` to `A` for the training generators, `K` to add KazLLM and Sherkala, `B` for leave-one-generator-out runs; `W` retrains the released classifier and saves its weights), `kernels/score_public_detectors.py` runs the public detectors. Fine-tuning ran on one 16 GB GPU, and zero-shot scoring placed the observer and performer models on two. The kernels read the benchmark file from `/kaggle/input` and can be pointed at a local copy.
5. **Analysis.** `analyze_det.py` computes every reported metric from the score files; `merge_zs.py` and `make_zs_delta.py` reuse zero-shot scores across benchmark versions; `surface_cues.py`, `ling.py` and `audit_agreement.py` produce the remaining tables; `dev_replication.py` repeats the fixes table on the development clusters with cross-fitted thresholds; `check_claims.py` checks the directional statements in the paper against the results.

To recompute the paper's detector results from the released scores, copy the dataset's `data/eval_index.jsonl` to `data/eval/eval.jsonl` and its `scores/` folder to the repository root, then run `scripts/reproduce_numbers.sh`. This reproduces every detector metric in the paper and the development-only replication exactly. The validity audits are in the dataset's `audit/` folder.

## Released detector

`detector/aina_kk_det.py` is the inference code of AINA-KK-Det, the same file that ships with the model on the Hugging Face Hub. `kernels/score_release_detector.py` runs it end to end over the benchmark (it needs the raw text, `data/eval/eval_raw.jsonl`, which `build_eval.py` writes), and `scripts/calibrate_release.py` turns those scores into the model's `calibration.json` and its measured error rates:

```
python3 scripts/calibrate_release.py scores/release_model/release_scores.jsonl scores/finetuned_A/det_meta.json results/release
```

The dataset's `scores/release_model/` folder holds the scores of the released model, so this step can be rerun without a GPU.

## Changes

- v1.2 (11 Oct 2026): `scripts/extra_analysis.py` (the released length rule applied to every input, all 44 stored zero-shot statistics, exact and threshold-inclusive intervals for the Kazakh-built subsets, McNemar tests), `scripts/dev_replication_deployed.py` (the development-only replication with the length rule), and `scripts/build_results_extra.py` (writes `paper/results_extra.json`, `paper/results_extra.tex` and the appendix Tables 8-9). `scripts/render_paper.py` now reads `paper/results_extra.json` for the length-rule rows of Tables 4 and 7. Order after `analyze_det.py` and `build_results.py`:

```
python3 scripts/extra_analysis.py scores/finetuned_A scores/finetuned_K scores/zero_shot results/extra.json
python3 scripts/dev_replication_deployed.py scores/finetuned_A scores/finetuned_K scores/zero_shot
python3 scripts/build_results_extra.py results/extra.json results/dev_replication_v2.json scores/release_model/rates.json paper results/results.json audit
python3 scripts/render_paper.py
```

- v1.1 (11 Oct 2026): training part `K` (KazLLM and Sherkala added to the training generators), the development-only replication (`dev_replication.py`), threshold-inclusive intervals in `analyze_det.py`, gray-zone rates on test-split sources only, and a claim checker that runs on the released files. Wikipedia text is labelled CC BY-SA 3.0, the license of 2022 revisions. v1.0 is unchanged.

## Citation

See `CITATION.cff`.

## License

Code: MIT. Data: see the dataset card; the human text keeps the license of its source.
