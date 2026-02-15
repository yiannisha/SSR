# SMART-SSR-CL Guide

This workspace contains the SSR research code under `XMUDeepLIT_SSR/`.
It is a fork/extension of LLaMA-Efficient-Tuning (LLaMA-Factory style) for continual learning experiments from the ACL 2024 SSR paper.

## 1) What this codebase is

Core training runner:
- `XMUDeepLIT_SSR/src/train_bash.py` -> calls `llmtuner.run_exp()`
- `XMUDeepLIT_SSR/src/llmtuner/tuner/tune.py` dispatches by `--stage`

Supported stages in this fork:
- `pt` (pretraining)
- `sft` (supervised fine-tuning)
- `sftrp` (SFT-style predict/eval used heavily for CL evaluation)
- `sftreg` (regularization CL: EWC/L2)
- `rm`, `ppo`, `dpo` (inherited from upstream framework)

SSR-specific additions live in:
- `XMUDeepLIT_SSR/custom/icl_gen/` (synthetic instance generation + labeling)
- `XMUDeepLIT_SSR/custom/niv2-c012/` (SuperNI filtering/splitting/KMeans subset creation)
- `XMUDeepLIT_SSR/src/scripts-ni-c012/` (experiment shell scripts)

## 2) What experiments are supported

### A. Continual learning on SuperNI categories (`ni-cus0.12`)
Task categories used in scripts:
- 10-task setup: `qa qg sa sum trans dsg expl para pe pos`
- 5-task setup: first five only (`qa qg sa sum trans`)

Experiment families (main paper-style):
- Single-task start (stage-1 CL init): `lora/sing/...`
- Non-rehearsal CL: `lora/cl...` with `cl_queue`
- RandSel rehearsal: scripts with `_rp`
- KMeansSel rehearsal: scripts with `_km20_rp`
- SSR rehearsal (self-synthesized + refined): scripts with `_iclgen_self`
- Queue variants: `cl`, `cl2`, `cl3` (different task orders / queues)

Representative script paths:
- Non-rehearsal: `XMUDeepLIT_SSR/src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue.3ep.bs32x1x1.lr2e-04.bf16.sh`
- RandSel: `XMUDeepLIT_SSR/src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_rp.3ep.bs32x1x1.lr2e-04.bf16.sh`
- KMeansSel: `XMUDeepLIT_SSR/src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_km20_rp.3ep.bs32x1x1.lr2e-04.bf16.sh`
- SSR: `XMUDeepLIT_SSR/src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh`

### B. Multi-task (joint) baselines
- `lora/all/...all...sh` (all categories)
- `lora/all/...all_5...sh` (5-category subset)

### C. Regularization-based CL baselines
- `--stage sftreg` with `--reg_cl_method ewc|l2`
- Example scripts in:
  - `XMUDeepLIT_SSR/src/scripts-ni-c012/lora/cl/llama2-7b-reg/`

### D. Synthetic data generation pipeline (SSR data creation)
Pipeline in README and custom scripts:
1. Generate ICL synthetic instances:
   - `custom/icl_gen/complete_param_nic010_cate.py`
2. Parse/filter generated outputs:
   - `custom/icl_gen/parser.py` (or alpaca variant scripts)
3. Select rehearsal subset:
   - Random: `custom/icl_gen/random_select.py`
   - KMeans: `custom/icl_gen/kmeans_self.py` (needs embeddings)
4. Refine synthetic outputs with current checkpoint:
   - `custom/icl_gen/label_param.py`

### E. Additional evaluation workflows
- CL eval in main scripts uses `--stage sftrp` and writes:
  - `all_results.json`
  - `generated_predictions.jsonl`
- MMLU:
  - `XMUDeepLIT_SSR/mmlu_test/evaluate_causal.py`
  - demo: `XMUDeepLIT_SSR/mmlu_test/mmlu_demo.sh`
- AlpacaEval helper:
  - `XMUDeepLIT_SSR/custom/alpaca_eval/alpaca_demo.sh`

## 3) Before running: critical path fixes

Most provided `.sh` scripts are hardcoded to old absolute paths like `/home/hjh/data/public/SSR`.
You must edit at least these variables in each script you use:
- `REPO_ROOT_DIR`
- `SRC_DIR`
- `MODEL_DIR`
- any hardcoded data output paths

For this workspace, use:
- `REPO_ROOT_DIR=/Users/yiannis/work/research/SMART-SSR-CL/XMUDeepLIT_SSR`

## 4) Environment setup

```bash
cd /Users/yiannis/work/research/SMART-SSR-CL/XMUDeepLIT_SSR
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- Default scripts use `--bf16 True`; you need BF16-capable GPU.
- If BF16 unsupported, switch scripts/commands to `--fp16 True` and remove `--bf16 True`.

## 5) Minimal way to run core CL experiments

### Step 1: train the first task checkpoint (required)

```bash
cd /Users/yiannis/work/research/SMART-SSR-CL/XMUDeepLIT_SSR
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --stage sft \
  --model_name_or_path /path/to/base/model \
  --do_train True \
  --overwrite_cache True \
  --finetuning_type lora \
  --template llama2 \
  --dataset_dir data \
  --dataset ni_c012_qa_train \
  --max_source_length 1024 \
  --max_target_length 512 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --lora_rank 8 \
  --lora_dropout 0.1 \
  --lora_target q_proj,v_proj \
  --output_dir saves/ni-c012/LLAMA2-7B-Chat/lora/qa/bs32x1x1-3ep-bf16 \
  --plot_loss True \
  --bf16 True
```

### Step 2: run a CL method for next task (`qg` example)

Use previous checkpoint from Step 1 via `--checkpoint_dir`.

Non-rehearsal:
```bash
--dataset ni_c012_qg_train
```

RandSel rehearsal:
```bash
--dataset ni_c012_qg_train,ni_c012_qa_train_smp01
```

KMeansSel rehearsal:
```bash
--dataset ni_c012_qg_train,ni_c012_qa_train_km20_smp01
```

SSR rehearsal:
```bash
--dataset ni_c012_qg_train,ni_c012_icl_gen_km20_self_cl_queue_llama2_7b_chat_qa
```

All of the above use the same `train_bash.py` pattern as Step 1, but add:
```bash
--checkpoint_dir <previous_task_ckpt>
```

### Step 3: evaluate each checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
  --stage sftrp \
  --model_name_or_path /path/to/base/model \
  --checkpoint_dir <ckpt_dir> \
  --overwrite_cache True \
  --predict_with_generate True \
  --finetuning_type lora \
  --template llama2 \
  --dataset_dir data \
  --dataset ni_c012_qg_eval \
  --max_source_length 1024 \
  --max_target_length 512 \
  --per_device_eval_batch_size 1 \
  --output_dir <ckpt_dir>/ni_c012_qg_eval \
  --do_predict True \
  --do_sample False \
  --bf16 True
```

Metrics are saved in:
- `<output_dir>/all_results.json`
- `<output_dir>/generated_predictions.jsonl`

## 6) Running the provided scripts (recommended for full sweeps)

After editing path variables, run directly:

- Single-task init:
```bash
bash src/scripts-ni-c012/lora/sing/llama2-7b-chat/llama2-7b-chat.lora.single.3ep.bs32x1x1.bf16.sh qa 0
```

- Non-rehearsal CL sweep:
```bash
bash src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue.3ep.bs32x1x1.lr2e-04.bf16.sh 0
```

- RandSel CL sweep:
```bash
bash src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_rp.3ep.bs32x1x1.lr2e-04.bf16.sh 0 01
```

- KMeansSel CL sweep:
```bash
bash src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_km20_rp.3ep.bs32x1x1.lr2e-04.bf16.sh 0 01
```

- SSR CL sweep:
```bash
bash src/scripts-ni-c012/lora/cl/llama2-7b-chat/llama2-7b-chat.lora.cl_queue_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh 0
```

- Queue variants:
```bash
bash src/scripts-ni-c012/lora/cl2/llama2-7b-chat/llama2-7b-chat.lora.cl_queue2_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh 0
bash src/scripts-ni-c012/lora/cl3/llama2-7b-chat/llama2-7b-chat.lora.cl_queue3_iclgen_self.3ep.bs32x1x1.lr2e-04.bf16.sh 0
```

## 7) How to run SSR synthetic data generation

Example flow (llama2-chat family):

1. Generate synthetic instances:
```bash
bash custom/icl_gen/scripts-ni-c012/llama2-7b-chat/ori-van.sh 0 "qa qg sa sum trans" 2 3 1.2
```

2. Parse/filter generated files:
- adapt and run `custom/icl_gen/parser.py` (has hardcoded paths)

3. Select subset:
- random: `custom/icl_gen/random_select.py`
- kmeans: compute embeddings first, then `custom/icl_gen/kmeans_self.py`

4. Refine synthetic outputs using current model:
```bash
bash custom/icl_gen/scripts-ni-c012/llama2-7b-chat/label-self.sh qa 0 "qg sa sum trans"
```

Important:
- These scripts also contain hardcoded paths and may need edits before use.
- Dataset keys referenced by CL scripts are pre-registered in `data/dataset_info.json`.

## 8) Data preprocessing from raw NI (if rebuilding dataset)

The `ni-cus0.12` split files are already present. If you need to rebuild:

1. Length filter raw NI task JSON:
- `custom/niv2-c012/1_length_fiiter.py`
2. Train/eval/extra split + sampled subsets:
- `custom/niv2-c012/2_split_and_random_selection.py`
3. (Optional) embedding + KMeans subset construction:
- `custom/niv2-c012/text2emb.py`
- `custom/niv2-c012/kmeans_selection.py`

## 9) Useful outputs to inspect

- Training logs/metrics per run: inside each `--output_dir`
- Evaluation metrics: `<output_dir>/all_results.json`
- Predictions: `<output_dir>/generated_predictions.jsonl`

## 10) Common gotchas

- Hardcoded paths in many research scripts are the #1 failure point.
- `dataset_info.json` key names must match `--dataset` exactly.
- `--predict_with_generate True` is required for `sftrp` prediction output.
- If output directory already exists and is non-empty, set unique `--output_dir` or `--overwrite_output_dir` as needed.

