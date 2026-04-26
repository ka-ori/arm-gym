# HuggingFace Jobs — MVP 50-step runs

## Run (retry) — `transformers-pytorch-gpu` image has no `curl` by default. First job `69ed72a7d2c8bd8662bcedbd` failed at `curl: command not found`. Use the `apt-get install curl` prefix below.

### Latest run

- **Job URL:** *(update after the replacement job is submitted)*  
- **Profile:** `ARMGYM_PROFILE=mvp` → **50 steps**  
- **Launcher:** `hf/job_run_mvp50_github_v6.sh` (curls `hf/run_hf_mvp.sh`, pins `v6_train.py` to **main** on GitHub for harness + G=4 + export manifest)  
- **Model repo (full run folder upload on success):** https://huggingface.co/ZDC-M01/arm-gym-mvp-50  

### Artifacts to pull for analysis

When training finishes, the job uploads the **entire** `runs/v6-mvp` output to the model repo, including:

| Artifact | Description |
|----------|-------------|
| `lora-adapter/` | PEFT LoRA weights + tokenizer |
| `config.json` | Frozen `Cfg` from the run |
| `export_manifest.json` | Config snapshot + checkpoint dir names |
| `log.csv` | Trainer `log_history` |
| `checkpoint-*` | HF Trainer checkpoint (MVP: ~step 50, `save_total_limit=1`) |
| `training_args.bin` / trainer state | If present under checkpoint |

Download (saves **everything** the job uploaded — LoRA, logs, manifest, checkpoints):

```bash
cd analysis/hf-mvp-50-runs
chmod +x download_from_hub.sh
HF_TOKEN=... ./download_from_hub.sh ZDC-M01/arm-gym-mvp-50 ./clone-mvp-50
```

(Use a [HF token](https://huggingface.co/settings/tokens) with read if the repo is private.)

### Re-run the same job

Set `HF_TOKEN` (write access to the model + dataset for wheel), then in Cursor with HF MCP, or from CLI:

`hf jobs run` with image `huggingface/transformers-pytorch-gpu`, flavor `a10g-large`, `timeout=3h`, and command:

```bash
curl -sSL https://raw.githubusercontent.com/ka-ori/arm-gym/main/hf/job_run_mvp50_github_v6.sh | bash
```
