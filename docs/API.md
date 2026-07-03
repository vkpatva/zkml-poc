# API usage guide

Pattern **A** — users call your hosted API. They never receive weight files or `model.py`.

## Model tiers

Two weight artifacts are trained from the same architecture; premium gets more epochs.

| Tier | Weights file | Training | API key env var |
|------|--------------|----------|-----------------|
| **free** | `artifacts/digit_mlp_free.pth` | 25 epochs | `FREE_API_KEY` |
| **premium** | `artifacts/digit_mlp_premium.pth` | 150 epochs | `PREMIUM_API_KEY` |

Send the matching key as header `X-API-Key`. The response includes `"tier": "free"` or `"premium"`.

## Prerequisites

- Both weight files exist (`python -m training.train`)
- ONNX exported (`python -m training.export_onnx`)
- EZKL set up for tiers you want to prove — see **[ZK.md](ZK.md)**
- Copy `.env.example` → `.env` and set keys for production

## Start the server

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m training.train
export FREE_API_KEY=dev-free-key
export PREMIUM_API_KEY=dev-premium-key
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If **no** keys are set, `/predict` uses the **free** tier with no auth (local dev only).

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Endpoints

### `GET /health`

No auth required.

```bash
curl http://localhost:8000/health
```

### `POST /predict`

**Headers**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | `application/json` |
| `X-API-Key` | If keys configured | `FREE_API_KEY` or `PREMIUM_API_KEY` value |

**Request body**

| Field | Type | Description |
|-------|------|-------------|
| `pixels` | `float[64]` | Row-major pixels in `[0, 1]` (`raw / 16.0`) |

**Response**

| Field | Type | Description |
|-------|------|-------------|
| `digit` | `int` | Predicted class 0–9 |
| `confidence` | `float` | Softmax probability of `digit` |
| `logits` | `float[10]` | Raw outputs |
| `tier` | `string` | `free` or `premium` |

**Free tier**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-free-key" \
  -d '{"pixels": [0.0, 0.125, 0.9375, 0.8125, 0.125, 0.0, 0.0, 0.0, 0.0, 0.4375, 1.0, 0.8125, 0.9375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}'
```

**Premium tier**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-premium-key" \
  -d '{"pixels": [0.0, 0.125, ...]}'
```

Example response:

```json
{
  "digit": 2,
  "confidence": 0.9999,
  "logits": [...],
  "tier": "premium"
}
```

### `POST /prove`

Generate a **ZK-SNARK proof** that inference is correct. Same `pixels` and tier auth as `/predict`. **Slow** (~30–120 s).

See **[ZK.md](ZK.md)** for the three scenarios (audit, model binding, private input).

**Request body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pixels` | `float[64]` | required | Same as `/predict` |
| `visibility` | `"public"` \| `"private"` | `"public"` | `public` = audit (input visible); `private` = hide pixels |

**Response**

| Field | Type | Description |
|-------|------|-------------|
| `digit` | `int` | `argmax` of EZKL logits |
| `logits` | `float[10]` | Circuit outputs (fixed-point) |
| `tier` | `string` | `free` or `premium` |
| `visibility` | `string` | Profile used |
| `model_id` | `string` | e.g. `digit_mlp_premium_public_audit` |
| `vk_hash` | `string` | SHA-256 of verifier key — pin to trust model M |
| `scenario` | `string` | `correct_inference_audit` or `private_input_proof` |
| `verified` | `bool` | Server verified proof immediately after generation |
| `proof` | `object` | Share with verifiers |

**Scenario 1 / 2 — public audit (premium)**

```bash
curl -X POST http://localhost:8000/prove \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-premium-key" \
  -d '{"pixels": [0.0, 0.125, ...], "visibility": "public"}'
```

**Scenario 3 — private input**

```bash
curl -X POST http://localhost:8000/prove \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-premium-key" \
  -d '{"pixels": [0.0, 0.125, ...], "visibility": "private"}'
```

### `POST /verify`

Verify a `proof` object from `/prove`. No API key required (verifier only needs local `vk.key` from setup).

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "proof": { "instances": [...], "proof": "0x..." },
    "tier": "premium",
    "visibility": "public"
  }'
```

| Field | Type | Description |
|-------|------|-------------|
| `verified` | `bool` | Cryptographic check passed |
| `model_id` | `string` | Expected model profile |
| `vk_hash` | `string` | Compare to your approved registry |
| `scenario` | `string` | Which visibility profile was used |

## Python client

```python
import requests

BASE = "http://localhost:8000"
headers = {"X-API-Key": "dev-premium-key"}

pixels = [0.0] * 64  # 8×8 normalized / 16.0
r = requests.post(f"{BASE}/predict", json={"pixels": pixels}, headers=headers)
data = r.json()
print(data["tier"], data["digit"], data["confidence"])
```

## Input rules

1. Exactly **64** floats (8×8, row-major).
2. Each value in **`[0, 1]`** — divide raw pixel (0–16) by `16.0`.

## Errors

| HTTP | Meaning |
|------|---------|
| `401` | Missing or invalid `X-API-Key` when keys are configured |
| `422` | Wrong pixel count or out-of-range values |
| `500` | Weights missing — run `python -m training.train` |
| `503` | EZKL not set up for tier/visibility — see [ZK.md](ZK.md) |

## What users do **not** get

- `artifacts/digit_mlp_free.pth` / `digit_mlp_premium.pth`
- `artifacts/*.onnx`
- `ezkl/**/pk.key`, `network.ezkl`
- `app/model.py`, `training/`, `proving/`

Users receive API responses and portable **proofs** (`proof`, `vk_hash`, `model_id`). See [ZK.md](ZK.md).
