# Zero-knowledge proofs (EZKL)

This project uses [EZKL](https://github.com/zkonduit/ezkl) to generate **ZK-SNARK proofs** that digit-classifier inference was computed correctly by a **specific model**, without sharing ONNX or PyTorch weights.

Full API details for `/prove` and `/verify` are in [API.md](API.md).

---

## Three proof scenarios

| # | Scenario | What you prove | EZKL profile | `model_id` example |
|---|----------|----------------|--------------|-------------------|
| **1** | **Correct inference (audit)** | These 10 logits are the correct output for this 64-pixel input | `visibility=public` | `digit_mlp_premium_public_audit` |
| **2** | **Specific model binding** | Same as #1, and the model is exactly tier **premium** (not free) | Different `vk_hash` per tier | `digit_mlp_premium_public_audit` vs `digit_mlp_free_public_audit` |
| **3** | **Private input** | Logits are correct; **pixels stay hidden** in the proof | `visibility=private` | `digit_mlp_premium_private_input` |

### Scenario 1 — Correct inference (audit / reuse)

**Statement:**  
*“For this 8×8 image, model M produced these 10 logits.”*

- Input pixels and logits are **public** in the proof.
- Anyone with `vk.key` (or pinned `vk_hash`) can verify without re-running the model.
- Use when you want a portable, auditable inference record.

**Maps to:** public visibility profile, `scenario: correct_inference_audit`.

### Scenario 2 — Specific model binding

**Statement:**  
*“This proof is valid only for approved model M (e.g. premium, not free).”*

- Each tier (`free`, `premium`) has its own EZKL setup → unique `vk_hash`.
- Verifiers pin an approved `vk_hash` registry entry; proofs for other tiers fail verification.
- Model weights and ONNX never leave the server.

**Maps to:** tier selected at prove time (`X-API-Key` or `--tier`), checked via `model_id` + `vk_hash` in the response.

### Scenario 3 — Private input

**Statement:**  
*“Some secret 64-pixel image was classified; these logits are correct for model M.”*

- Input is **private**; output logits are **public**.
- Verifier learns the prediction math is correct, not the raw pixels.

**Maps to:** `visibility: private` in request / `--visibility private`.

---

## What is shared vs kept private

| Artifact | Provider / server | Verifier / customer |
|----------|-------------------|---------------------|
| `artifacts/*.pth`, `*.onnx` | Keep private | Never needed |
| `ezkl/{tier}/{visibility}/pk.key` | Prover only | Never share |
| `ezkl/{tier}/{visibility}/vk.key` | Publish or share `vk_hash` | Needed to verify |
| `manifest.json` | Generated at setup | `model_id`, `vk_hash`, `scenario` |
| `proof` (from `/prove`) | Generated per request | **Share this** |
| `settings.json` | Same dir as `vk` | Needed to verify |

---

## Prerequisites

```bash
source .venv/bin/activate
pip install -r requirements.txt   # includes ezkl

python -m training.train
python -m training.export_onnx
```

ONNX must use a **fixed batch size** (`[1, 64]`) — already handled by `training/export_onnx.py`.

---

## One-time setup

Generate EZKL circuits and keys per **tier** and **visibility**:

```bash
# Minimum for demos (premium + public audit)
python -m proving.setup_ezkl --tier premium --visibility public

# Private-input scenario
python -m proving.setup_ezkl --tier premium --visibility private

# All combinations (free/premium × public/private) — takes several minutes
python -m proving.setup_ezkl --all
```

Artifacts are written under:

```
ezkl/
├── premium/
│   ├── public/
│   │   ├── settings.json
│   │   ├── network.ezkl
│   │   ├── kzg.srs
│   │   ├── vk.key          # share vk_hash with verifiers
│   │   ├── pk.key          # server only
│   │   └── manifest.json   # model_id, vk_hash, scenario
│   └── private/
│       └── ...
└── free/
    └── ...
```

`manifest.json` example:

```json
{
  "model_id": "digit_mlp_premium_public_audit",
  "tier": "premium",
  "visibility": "public",
  "vk_hash": "e2269191...",
  "onnx": "artifacts/digit_mlp_premium.onnx",
  "scenario": "correct_inference_audit"
}
```

**After retraining:** run `training.train` → `training.export_onnx` → `proving.setup_ezkl --force` for affected tiers. `vk_hash` will change.

---

## CLI — prove and verify

### Scenario 1 / 2 (public audit, premium)

```bash
python -m proving.prove_inference --tier premium --visibility public
```

### Scenario 3 (private input)

```bash
python -m proving.prove_inference --tier premium --visibility private
```

### Custom input file

```bash
python -m proving.prove_inference \
  --tier premium \
  --visibility public \
  --input artifacts/input_premium.json
```

### Verify proof on disk

```bash
python -m proving.prove_inference --tier premium --visibility public --verify-only
```

### Run tests

```bash
# Requires setup first; skipped automatically if ezkl/ not present
python -m tests.test_ezkl
```

---

## HTTP API

Start the server (after EZKL setup):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### `POST /prove` — generate proof

Same `pixels` as `/predict`. Tier from `X-API-Key`. Proving takes **~30–120 seconds**.

**Scenario 1 / 2 (public):**

```bash
curl -X POST http://localhost:8000/prove \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-premium-key" \
  -d '{"pixels": [0.0, 0.125, ...], "visibility": "public"}'
```

**Scenario 3 (private input):**

```bash
curl -X POST http://localhost:8000/prove \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-premium-key" \
  -d '{"pixels": [0.0, 0.125, ...], "visibility": "private"}'
```

Response fields:

| Field | Meaning |
|-------|---------|
| `digit` | `argmax(logits)` from EZKL witness |
| `logits` | Fixed-point circuit outputs (≈ PyTorch) |
| `model_id` | Binds proof to one compiled model profile |
| `vk_hash` | Pin this to trust a specific model M |
| `scenario` | `correct_inference_audit` or `private_input_proof` |
| `verified` | Server ran `ezkl.verify` immediately after proving |
| `proof` | Portable JSON — send to any verifier with matching `vk` |

### `POST /verify` — check a proof

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "proof": { "...": "paste proof object from /prove" },
    "tier": "premium",
    "visibility": "public"
  }'
```

Returns `verified: true` if the proof is valid for that tier and visibility profile.

---

## Typical flows

### Audit — share proof publicly (Scenario 1)

1. Provider runs `setup_ezkl` once.
2. Client calls `POST /prove` with `visibility: public`.
3. Client shares `proof`, `model_id`, `vk_hash` with auditor.
4. Auditor calls `POST /verify` or runs `proving.prove_inference --verify-only`.

### Model binding (Scenario 2)

1. Provider publishes approved list: `model_id` → `vk_hash`.
2. Client proves with premium API key.
3. Verifier checks `vk_hash` matches registry **and** `ezkl.verify` passes.

### Private customer data (Scenario 3)

1. Provider runs `setup_ezkl --visibility private`.
2. Client calls `POST /prove` with `visibility: private`.
3. Client shares only `proof` + public logits/digit — pixels not revealed.

---

## Limitations

### Cryptographic / model

| Limitation | Detail |
|------------|--------|
| **No softmax / confidence** | Circuit proves raw **logits** only. `/predict` `confidence` is PyTorch softmax, not proven. |
| **Fixed-point arithmetic** | EZKL uses fixed-point math; logits differ slightly from float32 (~0.01). `argmax` usually agrees. |
| **Fixed batch size** | ONNX input is `[1, 64]` only — one image per proof. |
| **Setup per model version** | Retrain or change ONNX → re-run `setup_ezkl`; `vk_hash` changes. |
| **Setup per visibility** | Public and private input need **separate** EZKL setups. |

### Performance

| Limitation | Detail |
|------------|--------|
| **Slow proving** | ~30–120 s per proof on CPU for this small MLP (hardware dependent). |
| **Slow verification** | ~10–20 s per verify in Python bindings. |
| **Not for real-time `/predict`** | Use `/predict` for latency; `/prove` for audit trails. |

### Trust / scope

| Limitation | Detail |
|------------|--------|
| **Proves circuit correctness** | Proof shows ONNX→circuit execution is correct, not that off-chain input data is “truthful”. |
| **`vk` trust** | Verifiers must trust `vk_hash` came from the approved model (audit / registry). |
| **No provider attestation** | Proof does not by itself prove *who* ran inference (needs TEEs/signing separately). |
| **SRS in this POC** | Setup uses `gen_srs` for local dev. Production should use `ezkl.get_srs` with the official ceremony. |

### Operations

| Limitation | Detail |
|------------|--------|
| **Large local artifacts** | `ezkl/` is gitignored; each developer runs setup locally. |
| **CI** | Full EZKL tests are slow; `test_ezkl` skips if setup is missing. |
| **Tier binding test** | Requires `python -m proving.setup_ezkl --all` for free vs premium `vk_hash` comparison. |

---

## Module reference

| Module | Purpose |
|--------|---------|
| `proving/setup_ezkl.py` | One-time EZKL compile + keys |
| `proving/prove_inference.py` | CLI prove / verify |
| `proving/ezkl_core.py` | `setup_tier`, `prove_pixels`, `verify_proof_payload` |
| `proving/ezkl_paths.py` | Paths, `model_id`, `vk_hash`, manifests |
| `app/main.py` | `POST /prove`, `POST /verify` |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `503 EZKL not set up` | Run `python -m proving.setup_ezkl --tier <tier> --visibility <vis>` |
| `Undetermined symbol: Sym0` | Re-export ONNX without dynamic batch: `python -m training.export_onnx` |
| Digit matches PyTorch but logits differ | Expected fixed-point drift; check `max_logit_diff < 0.01` |
| Verify fails after retrain | Re-run `setup_ezkl --force`; update pinned `vk_hash` |
| `test_ezkl` skipped | Run setup for premium/public first |
