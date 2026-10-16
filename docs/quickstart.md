# Quickstart: Running on UVA HPC (Rivanna)

Step-by-step commands for the cross-modal probe pipeline. Gates from [design.md](../openspec/changes/add-cross-modal-probe/design.md) (D9) are marked **GATE**; stop and check before continuing.

Compute: account `ai-safety-hackathon-rig`, partition `gpu`, one A100, 16 CPUs, 128 GB, at most 4 h per job. **Never submit an `sbatch` job without the team agreeing first.**

## 0. Connect

```bash
# Off-grounds: connect to the UVA Anywhere VPN first
ssh -Y esd4uq@login.hpc.virginia.edu
```

## 1. Get the repo and create directories

```bash
cd ~
git clone git@github.com:<your-github-user>/xmodal-injection-probe.git
cd xmodal-injection-probe
mkdir -p logs data /scratch/esd4uq/xmprobe /scratch/esd4uq/hf_cache
```

`logs/` must exist before `sbatch`, because SLURM won't create the output directory.

## 2. Create the project environment (never modify `torch311`)

```bash
module load miniforge
eval "$(conda shell.bash hook)"
conda create -y -n xmprobe --clone torch311
conda activate xmprobe
pip install -U transformers accelerate huggingface_hub
pip install -e ".[dev]"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
pytest   # CPU unit tests, a few seconds
```

## 3. Data: LLMail-Inject → `data/pairs.jsonl` (login node, CPU only)

```bash
export HF_HOME=/scratch/esd4uq/hf_cache
huggingface-cli download microsoft/llmail-inject-challenge --repo-type dataset \
  --local-dir /scratch/esd4uq/hf_cache/llmail-inject-challenge \
  --include "data/raw_submissions_phase1.jsonl" "data/emails_for_fp_tests.json" "data/scenarios.json" \
            "data/system_prompt.json" "data/levels_descriptions.json" "data/objectives_descriptions.json"
python scripts/convert_llmail.py --config configs/qwen35_2b.yaml
cat data/pairs.report.json
python -m xmprobe.cli validate_data --config configs/qwen35_2b.yaml
```

**GATE**: validation passes. Expected counts (measured 2026-10-16): about 173 `main` and 58 `probe_evading` pairs. The raw file is about 1.7 GB and stays on scratch.

## 4. Interactive GPU session, model download, toolchain check — **GATE**

```bash
srun -A ai-safety-hackathon-rig -p gpu --time=04:00:00 --nodes=1 --ntasks-per-node=1 \
     --cpus-per-task=16 --mem=128G --gres=gpu:a100:1 --pty bash
module load miniforge && eval "$(conda shell.bash hook)" && conda activate xmprobe
cd ~/xmodal-injection-probe
export HF_HOME=/scratch/esd4uq/hf_cache

# R1: confirm these IDs on huggingface.co first
huggingface-cli download Qwen/Qwen3.5-0.8B
huggingface-cli download Qwen/Qwen3.5-2B
# fallback, only if the check below fails:
# huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct

python scripts/check_toolchain.py configs/qwen35_0p8b.yaml
python scripts/check_toolchain.py configs/qwen35_2b.yaml
```

**GATE**: `GATE: PASS` for both. That requires hidden states with `num_hidden_layers + 1` entries and `thinking leaked: False`. Copy the printed revision into `model.revision` in the matching config (R1). If this fails and can't be fixed in about 15 minutes, use `configs/qwen25vl_3b.yaml` for everything below.

## 5. Smoke run (same interactive session) — **GATE**

```bash
export HF_HUB_OFFLINE=1
python -m xmprobe.cli all --config configs/qwen35_0p8b.yaml --limit 10
ls /scratch/esd4uq/xmprobe/runs/qwen35_0p8b_v1/
```

**GATE**: every stage writes valid outputs in under 20 minutes. `all` stops with exit code 3 at a failed gate. With only 10 pairs the OCR gate fails by design (n < 30), so continue the smoke run stage by stage:

```bash
for s in generate extract probe report; do python -m xmprobe.cli $s --config configs/qwen35_0p8b.yaml --limit 10; done
```

## 5b. Demo: 100 pairs on Qwen3.5-2B (same session; informational, not a gate)

```bash
python -m xmprobe.cli demo --config configs/qwen35_2b.yaml
cat /scratch/esd4uq/xmprobe/runs/qwen35_2b_v1/demo/reports/headline.md
```

Demo numbers are a signal check (AUROC CIs roughly ±0.05–0.10), not final results. Check `stages.demo.wall_seconds` in `run_manifest.json`; the target is under 900 s (SC-007). Exit the interactive session (`exit`) before submitting batch jobs.

## 6. Full run on Qwen3.5-2B (batch; agree on the submission first)

The job scripts are `slurm/gpu_stage.sbatch` and `slurm/cpu_probe.sbatch`. Dry-run a submission first:

```bash
cd ~/xmodal-injection-probe
CFG=configs/qwen35_2b.yaml
RUN=/scratch/esd4uq/xmprobe/runs/qwen35_2b_v1
sbatch --test-only slurm/gpu_stage.sbatch $CFG render

# 6a. render (seconds, CPU — fine on the login node), then the OCR gate on GPU
python -m xmprobe.cli render --config $CFG
sbatch slurm/gpu_stage.sbatch $CFG ocr_check
cat $RUN/ocr_summary.json        # after the job finishes
```

**GATE (OCR)**: `"pass": true` (median CER ≤ 0.10, n ≥ 30). If it fails, set `render.font_size: 28`, then re-run `render --force` and `ocr_check --force` once. If it still fails, switch to the fallback model.

```bash
# 6b. generation -> extraction -> probes + report, chained
j1=$(sbatch --parsable slurm/gpu_stage.sbatch $CFG generate)
j2=$(sbatch --parsable --dependency=afterok:$j1 slurm/gpu_stage.sbatch $CFG extract)
sbatch --dependency=afterok:$j2 slurm/cpu_probe.sbatch $CFG
# if the account cannot use the `standard` partition:
# sbatch -p gpu --gres=gpu:a100:1 --dependency=afterok:$j2 slurm/cpu_probe.sbatch $CFG
```

**GATE (behavior)**: after `generate`, check `$RUN/behavior_summary.csv`. If the `text_injected` target rate is below 5%, obey probes are skipped automatically; the present probe is still the main result.

## 7. Monitor

```bash
squeue -u esd4uq
tail -f logs/xmprobe-gpu-<jobid>.out
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,AllocTRES%40
```

## 8. Read and save results — **GATE (P1)**

```bash
cat $RUN/reports/headline.md
```

**GATE (P1)**: each method's P1 AUROC ≥ 0.80 at its headline layer. A method that fails has its P2/P3 rows labelled "not interpretable (P1)".

Scratch is not permanent, so copy the summaries into the repo:

```bash
mkdir -p results/qwen35_2b_v1
cp -r $RUN/reports/* results/qwen35_2b_v1/
cp $RUN/behavior_summary.csv $RUN/ocr_summary.json $RUN/run_manifest.json $RUN/probes/*.csv results/qwen35_2b_v1/
git add results/qwen35_2b_v1
git commit -m "results: qwen35_2b_v1"
git push
```
