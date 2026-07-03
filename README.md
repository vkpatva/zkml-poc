# zkml-poc

A minimal proof-of-concept for **zero-knowledge machine learning (ZKML)**: train a small digit classifier, serve it via a **hosted API (Pattern A)**, and later export to ONNX for proving with [EZKL](https://github.com/zkonduit/ezkl).

The model uses only `Linear` and `ReLU` layers so the graph stays circuit-friendly.

## What it does

- Trains an MLP on scikit-learn's 8×8 handwritten digits (0–9)
- Saves **two** weight tiers: `digit_mlp_free.pth` (25 epochs) and `digit_mlp_premium.pth` (150 epochs)
- Routes `X-API-Key` → free or premium model at inference time
- Golden-sample test before ONNX / ZKML

## Project structure

```
zkml-poc/
├── app/                      # Hosted API (Pattern A)
│   ├── main.py               # FastAPI app — POST /predict
│   ├── inference.py          # Load weights + run inference
│   ├── model.py              # DigitMLP architecture (private on server)
│   ├── schemas.py            # Request/response models
│   └── config.py             # Tier paths + API key mapping
├── training/
│   └── train.py              # Train free + premium weights
├── tests/
│   ├── test_pipeline.py      # Golden-sample sanity check
│   └── fixtures/
│       ├── test_input.json
│       ├── expected_output.json
│       └── test_sample_preview.png
├── artifacts/
│   ├── digit_mlp_free.pth     # Free tier weights
│   └── digit_mlp_premium.pth  # Premium tier weights
├── docs/
│   └── API.md                # How end users call the API
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+ (tested on 3.12)
- ~2 GB disk for PyTorch

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

## Pattern A — what stays private

| Private (server) | Public (users) |
|------------------|----------------|
| `artifacts/*.pth` | `POST /predict` + tier in response |
| `app/model.py` | Request/response schema in docs |
| `training/` | `GET /health` |

## Roadmap

- [x] Step 1 — Circuit-friendly `DigitMLP`
- [x] Step 2 — Train + golden test
- [x] Step 2b — Pattern A HTTP API
- [ ] Step 3 — ONNX export
- [ ] Step 4 — EZKL proofs

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `python` not found | Use `python3` or activate `.venv` |
| Weights not found | `python -m training.train` |
| `test_pipeline` FAIL | Check `/16.0` normalization and retrain |
| `401` on `/predict` | Send valid `X-API-Key` matching `FREE_API_KEY` or `PREMIUM_API_KEY` |

## License

Add your license here (e.g. MIT, Apache-2.0).
