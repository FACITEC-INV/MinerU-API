# MinerU OCR Demo

This tool focuses on integrating the selected OCR model into a user-friendly Python tool, enabling seamless data extraction from Paraguayan identity documents.

## Requirements

- NVIDIA GPU with CUDA support

## Installation (Recommended with uv)

1. (Optional but recommended) Create and activate a virtual environment:

```powershell
uv venv .mineru
source .mineru/bin/activate
```

2. Install dependencies with `uv`:

```powershell
uv pip install "mineru-vl-utils[vllm]" fastapi uvicorn pillow pandas fuzzywuzzy
```

## Project Files

- `app.py`: Local script that reads `input/image.jpg`, extracts text, and parses it.
- `api.py`: FastAPI service with OCR + parsing endpoint.
- `benchmark.py`: FastAPI service for load-time and inference benchmark metrics.
- `ci_parser.py`: Fuzzy parser that maps OCR text into structured fields.

## Quick Start (Local Script)

Run:

```powershell
python app.py
```

What it does:

- Loads the model `opendatalab/MinerU2.5-2509-1.2B`
- Reads `input/image.jpg`
- Extracts OCR blocks
- Joins text content
- Parses output into fields such as:
  - `apellidos`
  - `nombres`
  - `fecha_nacimiento`
  - `lugar_nacimiento`
  - `fecha_vencimiento`
  - `sexo`
  - `numero_documento`

## Run the OCR API

Start server:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoint:

- `POST /ocr/` with an image file (`multipart/form-data`, field name: `file`)

Example request:

```powershell
curl -X POST "http://127.0.0.1:8000/ocr/" -F "file=@input/image.jpg"
```

Example response:

```json
{
  "raw_text": "APELLIDOS\nGONZALEZ\nNOMBRES\nMARIA ELENA\nFECHA DE NACIMIENTO\n12/05/1992\nLUGAR DE NACIMIENTO\nASUNCION\nSEXO\nF\nVENCIMIENTO\n12/05/2032\nDOCUMENTO 4567890",
  "json": {
    "apellidos": "GONZALEZ",
    "nombres": "MARIA ELENA",
    "fecha_nacimiento": "12/05/1992",
    "lugar_nacimiento": "ASUNCION",
    "fecha_vencimiento": "12/05/2032",
    "sexo": "F",
    "numero_documento": "4567890"
  }
}
```

## Run the Benchmark API

Start server:

```powershell
uvicorn benchmark:app --host 0.0.0.0 --port 8001
```

Endpoint:

- `POST /benchmark/` with an image file (`multipart/form-data`, field name: `file`)

Example request:

```powershell
curl -X POST "http://127.0.0.1:8001/benchmark/" -F "file=@input/image.jpg"
```

Example response:

```json
{
  "load_time_s": 8.374,
  "gpu_load_peak_mb": 3210.42,
  "inference_time_s": 0.684,
  "gpu_inference_peak_mb": 3428.17
}
```

## Notes

- The first model load can take time and use significant GPU memory.
- If your environment cannot use CUDA, behavior and performance may vary.
