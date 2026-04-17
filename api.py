from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from vllm import LLM
from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
import pandas as pd
from ci_parser import CIParser

# -----------------------------
# 🔹 Inicializar API
# -----------------------------
app = FastAPI(title="MinerU OCR API")

llm = None
client = None
parser = None

# -----------------------------
# 🔹 Startup Event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global llm, client, parser

    llm = LLM(
        model="opendatalab/MinerU2.5-2509-1.2B",
        logits_processors=[MinerULogitsProcessor]
    )

    client = MinerUClient(
        backend="vllm-engine",
        vllm_llm=llm
    )

    parser = CIParser()

    print("✅ MinerU + parser inicializados correctamente")

# -----------------------------
# 🔹 Shutdown Event
# -----------------------------
@app.on_event("shutdown")
async def shutdown_event():
    global llm, parser, client

    if llm:
        llm.shutdown()
        llm = None
        print("🛑 LLM MinerU cerrado")

    if parser:
        parser.close()
        parser = None
        print("🛑 CIParser cerrado")

    client = None

# -----------------------------
# 🔹 Endpoint OCR
# -----------------------------
@app.post("/ocr/")
async def ocr_image(file: UploadFile = File(...)):
    global client, parser

    if client is None or parser is None:
        return JSONResponse({"error": "Modelo no inicializado"}, status_code=500)

    try:
        # Leer imagen
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Ejecutar extracción MinerU
        extracted_blocks = client.two_step_extract(image)

        # Convertir a DataFrame y filtrar contenido textual
        df = pd.DataFrame(extracted_blocks)
        textos = df[df['content'].notnull()]['content']
        resultado = "\n".join(textos)

        # Parsear
        parsed = parser.parse(resultado)

        return {
            "raw_text": resultado,
            "json": parsed
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)