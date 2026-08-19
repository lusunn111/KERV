# Drafter training

KERV uses a frozen OpenVLA policy as the verifier and trains a lightweight
one-layer drafter to predict its hidden states and action-token distribution.
The released repository does not contain model weights or generated training
samples.

## 1. Prepare the base policy and RLDS data

Prepare an OpenVLA checkpoint fine-tuned on `libero_goal`, and convert the
corresponding demonstrations to the OpenVLA RLDS layout. The checkpoint must
contain its Hugging Face configuration, tokenizer, image processor and
`dataset_statistics.json`.

## 2. Generate teacher samples

The teacher is frozen. For each demonstration step, the generation script
stores the prompt IDs, image tensor, verifier hidden states and predicted
action tokens used as drafter supervision.

```bash
export PYTHONPATH="$PWD:$PWD/openvla"

python training/generate_drafter_data.py \
  --base-model /path/to/openvla-libero-goal \
  --data-root /path/to/modified_libero_rlds \
  --dataset-name libero_goal_no_noops \
  --output-dir /path/to/kerv_drafter_data
```

Use `--max-samples` for a small pipeline check before generating the complete
dataset. Generated `.ckpt` samples are intentionally ignored by Git.

## 3. Train the drafter

The reference recipe uses a one-layer Llama-style drafter, FP16 DeepSpeed
ZeRO-2, micro-batch size 4, AdamW with learning rate `5e-5`, 2,000 warm-up
steps, hidden-state Smooth-L1 loss and action-distribution distillation loss.
The frozen verifier head converts teacher and drafter hidden states into the
same action-token space.

```bash
deepspeed --master_port 23333 training/train_drafter.py \
  --base-model /path/to/openvla-libero-goal \
  --data-dir /path/to/kerv_drafter_data \
  --output-dir /path/to/kerv-drafter \
  --drafter-config training/drafter_config.json \
  --epochs 200 \
  --deepspeed_config training/deepspeed_zero2.json
```

The inference loader expects the selected checkpoint directory to contain:

```text
kerv-drafter/
├── config.json
└── pytorch_model.bin
```

Copy `training/drafter_config.json` to `config.json` and use the exported
16-bit `pytorch_model.bin` from the selected epoch. Select the epoch with a
held-out LIBERO validation split; do not choose it from inference latency.

## 4. Validate before release

Run the BF16 safe configuration first. A candidate drafter is accepted only
after checking action tokens, accepted lengths, selected paths, Kalman
branches and final environment actions against the reference implementation.
Report task success rate separately from the system-level latency benchmark.
