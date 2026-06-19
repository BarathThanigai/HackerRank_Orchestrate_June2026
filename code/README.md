# Multi-Modal Evidence Review

Production-style Python pipeline for the HackerRank Orchestrate June 2026
challenge. It extracts multilingual claims, independently analyzes every local
image with a vision model, applies evidence rules and user-history risk context,
and writes the exact required `output.csv` schema.

## Architecture

1. `claim_extractor.py` extracts claimed parts and issues from English, Hindi,
   Hinglish, and Spanish conversations.
2. `image_analyzer.py` sends each image independently to a pluggable vision
   backend and requires structured visual observations. Ollama/Qwen2.5-VL is the
   default free local backend; OpenAI remains available as an optional backend.
3. `evidence_validator.py` applies `evidence_requirements.csv`.
4. `risk_assessor.py` combines image-quality/authenticity flags with history flags.
   History never overrides visual evidence.
5. `decision_engine.py` deterministically returns supported, contradicted, or not
   enough information.
6. `main.py` validates and writes the exact output schema.

Images are primary evidence. Text inside an image is treated as untrusted content.

## Setup

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r code\requirements.txt
```

Unix/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

No paid API key is required for the default local setup.

## Free local vision setup with Ollama

Install Ollama:

- Windows/macOS: download and install from <https://ollama.com/download>
- Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama, then download the default vision model:

```bash
ollama pull qwen2.5vl:7b
```

If that tag is unavailable on your Ollama installation, use another local vision
model such as `llava:7b` and set `OLLAMA_MODEL` to that model name.

PowerShell configuration:

```powershell
$env:VISION_BACKEND="ollama"
$env:OLLAMA_MODEL="qwen2.5vl:7b"
$env:OLLAMA_URL="http://localhost:11434"
```

Unix/macOS configuration:

```bash
export VISION_BACKEND=ollama
export OLLAMA_MODEL=qwen2.5vl:7b
export OLLAMA_URL=http://localhost:11434
```

Once the model is downloaded, image analysis and `output.csv` generation run
locally without paid API usage. Environment variables can also be placed in a
local `.env` file, but never commit secrets.

## Generate predictions

```bash
python code/main.py
```

This reads `dataset/claims.csv` and writes `output.csv` plus
`output.telemetry.json`.

Useful options:

```bash
python code/main.py --input dataset/claims.csv --output output.csv
python code/main.py --model qwen2.5vl:7b --refresh-cache --verbose
```

Responses are cached under `code/.cache/`. The cache key includes image bytes,
the extracted claim, backend, model, and prompt version.

## Evaluate

```bash
python code/evaluation/main.py
```

This generates:

- `code/evaluation/sample_predictions.csv`
- `code/evaluation/metrics.json`
- `code/evaluation/evaluation_report.md`

Expected labels are loaded only after predictions have been generated.

## Failure behavior

- Missing or corrupt images yield a conservative review result.
- API failures are retried and isolated to the affected image.
- If no image can be analyzed, status is `not_enough_information` with
  `manual_review_required`.
- Every output row is enum- and schema-validated.
- One failed claim does not terminate the batch.

## Configuration

See `.env.example`. Defaults are centralized in `config.py`. Cost figures are
zero for the default Ollama backend after the model has been downloaded.

Common settings:

```text
VISION_BACKEND=ollama
OLLAMA_MODEL=qwen2.5vl:7b
OLLAMA_URL=http://localhost:11434
```

Optional OpenAI backend:

```bash
pip install openai
export VISION_BACKEND=openai
export OPENAI_API_KEY=...
export OPENAI_VISION_MODEL=gpt-5.5
```
