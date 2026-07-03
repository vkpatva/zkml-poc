# zkml-poc

A minimal proof-of-concept for **zero-knowledge machine learning (ZKML)**: train a small digit classifier, serve it via a **hosted API (Pattern A)**, export to ONNX, and prove inference with [EZKL](https://github.com/zkonduit/ezkl).

The model uses only `Linear` and `ReLU` layers so the graph stays circuit-friendly.

## What it does

- Trains an MLP on scikit-learn's 8×8 handwritten digits (0–9)
- Saves **two** weight tiers: `digit_mlp_free.pth` (25 epochs) and `digit_mlp_premium.pth` (150 epochs)
- Exports each tier to ONNX (`digit_mlp_{tier}.onnx`) with EZKL-ready sample inputs
- Routes `X-API-Key` → free or premium model at inference time
- Golden-sample tests for PyTorch inference, ONNX parity, and optional EZKL proofs

## Project structure

```
zkml-poc/
├── app/                      # Hosted API (Pattern A)
│   ├── main.py               # FastAPI — /predict, /prove, /verify
│   ├── inference.py          # Load weights + run inference
│   ├── model.py              # DigitMLP architecture (private on server)
│   ├── schemas.py            # Request/response models
│   └── config.py             # Tier paths + API key mapping
├── training/
│   ├── train.py              # Train free + premium weights
│   └── export_onnx.py        # Export .pth → .onnx + verify parity
├── proving/
│   ├── setup_ezkl.py         # One-time EZKL setup per tier + visibility
│   ├── prove_inference.py    # CLI prove / verify
│   ├── ezkl_core.py          # setup_tier, prove_pixels, verify_proof_payload
│   └── ezkl_paths.py         # Paths, model_id, vk_hash
├── tests/
│   ├── test_pipeline.py      # Golden-sample sanity check
│   ├── test_ezkl.py          # EZKL prove/verify (requires local setup)
│   └── fixtures/
│       ├── test_input.json
│       ├── expected_output.json
│       └── test_sample_preview.png
├── artifacts/
│   ├── digit_mlp_free.pth       # Free tier PyTorch weights
│   ├── digit_mlp_premium.pth    # Premium tier PyTorch weights
│   ├── digit_mlp_free.onnx      # Free tier ONNX graph (for EZKL / onnxruntime)
│   ├── digit_mlp_premium.onnx   # Premium tier ONNX graph
│   ├── input_free.json          # Sample input (EZKL witness format)
│   └── input_premium.json       # Sample input (EZKL witness format)
├── docs/
│   ├── API.md                # HTTP API (/predict, /prove, /verify)
│   └── ZK.md                 # EZKL scenarios, setup, limitations
├── ezkl/                     # Generated locally (gitignored)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+ (tested on 3.12)
- ~2 GB disk for PyTorch; extra space for EZKL artifacts under `ezkl/`

On Ubuntu/Debian use `python3` if `python` is not installed.

## Quick start (developers)

```bash
git clone <your-repo-url>
cd zkml-poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
python -m training.train
```

### Verify

```bash
python -m tests.test_pipeline
```

### Export ONNX

After training, export both tiers to ONNX and verify they match PyTorch on the golden sample:

```bash
python -m training.export_onnx
```

This writes `artifacts/digit_mlp_{tier}.onnx` and `artifacts/input_{tier}.json` for each tier.

### EZKL setup (one-time per tier + visibility)

See **[docs/ZK.md](docs/ZK.md)** for scenarios, limitations, and full guide.

```bash
python -m proving.setup_ezkl --tier premium --visibility public
python -m proving.setup_ezkl --tier premium --visibility private
```

### Prove (CLI)

```bash
python -m proving.prove_inference --tier premium --visibility public
python -m tests.test_ezkl   # optional; skips if ezkl/ not set up
```

### Run API

```bash
export FREE_API_KEY=dev-free-key
export PREMIUM_API_KEY=dev-premium-key
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger UI.

## API usage (end users)

See **[docs/API.md](docs/API.md)** for full details: endpoints, curl examples, Python/JS clients, and input format.

Minimal example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-premium-key" \
  -d '{"pixels": [0.0, 0.125, ...]}'
```

Set `FREE_API_KEY` / `PREMIUM_API_KEY` in `.env` on the server; clients send the matching `X-API-Key`.

## Model architecture

```
Input (64) → Linear(64→32) → ReLU → Linear(32→16) → ReLU → Linear(16→10) → logits
```

| Design choice | Reason |
|---------------|--------|
| No BatchNorm / Dropout | Harder in ZK circuits |
| No final Softmax | `argmax(logits)` suffices; softmax is expensive to prove |
| Normalization `/ 16.0` | Maps pixels (0–16) to `[0, 1]` |

## ONNX export

`training/export_onnx.py` loads each tier's `.pth` weights, traces `DigitMLP` with `torch.onnx.export`, and checks parity with onnxruntime on the golden fixture.

| Artifact | Description |
|----------|-------------|
| `digit_mlp_{tier}.onnx` | Portable model graph + embedded weights. Input: `input` `[batch, 64]`. Output: `logits` `[batch, 10]`. |
| `input_{tier}.json` | Sample pixels in EZKL format: `{"input_data": [[64 floats]]}`. |

The API still runs PyTorch (`.pth`) at inference time. ONNX is the interchange format for EZKL proving and optional onnxruntime deployment. Softmax is applied only in the API for confidence display — the ONNX graph exports raw logits.

Re-export after retraining:

```bash
python -m training.train
python -m training.export_onnx
```

## ZK proofs (EZKL)

Three scenarios are supported — see **[docs/ZK.md](docs/ZK.md)**:

1. **Correct inference** — auditable proof that logits are right (`visibility: public`)
2. **Specific model** — proof binds to tier via unique `vk_hash` (`X-API-Key` → tier)
3. **Private input** — prove without revealing pixels (`visibility: private`)

```bash
curl -X POST http://localhost:8000/prove \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-premium-key" \
  -d '{"pixels": [0.0, 0.125, ...], "visibility": "public"}'
```

## Pattern A — what stays private

| Private (server) | Public (users) |
|------------------|----------------|
| `artifacts/*.pth`, `*.onnx` | `POST /predict`, `/prove` responses |
| `ezkl/**/pk.key`, `network.ezkl` | `vk_hash`, `proof`, `model_id` |
| `app/model.py` | Request/response schema in docs |
| `training/`, `proving/` | `GET /health` |

## Roadmap

- [x] Step 1 — Circuit-friendly `DigitMLP`
- [x] Step 2 — Train + golden test
- [x] Step 2b — Pattern A HTTP API
- [x] Step 3 — ONNX export
- [x] Step 4 — EZKL proofs

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `python` not found | Use `python3` or activate `.venv` |
| Weights not found | `python -m training.train` |
| ONNX export fails | Run `pip install -r requirements.txt` (needs `onnx`, `onnxscript`, `onnxruntime`) |
| ONNX verify FAIL | Re-run `python -m training.train` then `python -m training.export_onnx` |
| `test_pipeline` FAIL | Check `/16.0` normalization and retrain |
| `401` on `/predict` | Send valid `X-API-Key` matching `FREE_API_KEY` or `PREMIUM_API_KEY` |
| `503` on `/prove` | Run `python -m proving.setup_ezkl --tier <tier> --visibility <vis>` |
| `test_ezkl` skipped | EZKL not set up locally — see docs/ZK.md |

## License

Add your license here (e.g. MIT, Apache-2.0).
