#!/usr/bin/env bash
# 200-step GRPO + LoRA. Overrides hf/PROFILE when sourced before python3 hf/v6_train.py
export ARMGYM_PROFILE=long
echo "ARMGYM_PROFILE=long → 200 steps, out runs/v6-200, hub ZDC-M01/arm-gym-train-200"
