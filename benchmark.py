from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from vllm import LLM
from mineru_vl_utils import MinerUClient, MinerULogitsProcessor
import torch
import time
import threading

# -----------------------------
# 🔹 Inicializar API
# -----------------------------
app = FastAPI(title="MinerU OCR Benchmark API")

llm = None
client = None
gpu_before_load = 0
gpu_load_peak_mb = 0
load_time = 0

# -----------------------------
# 🔹 Startup Event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global llm, client, gpu_before_load, gpu_load_peak_mb, load_time

    # Limpiar memoria GPU antes de cargar el modelo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_before, total_before = torch.cuda.mem_get_info()
        gpu_before_load = total_before - free_before
    else:
        gpu_before_load = 0

    # Tiempo de carga del modelo
    start_load = time.time()
    llm = LLM(
        model="opendatalab/MinerU2.5-2509-1.2B",
        logits_processors=[MinerULogitsProcessor]
    )
    load_time = time.time() - start_load

    # Medir memoria usada durante la carga
    if torch.cuda.is_available():
        free_after, total_after = torch.cuda.mem_get_info()
        gpu_after_load = total_after - free_after
        gpu_load_peak_mb = (gpu_after_load - gpu_before_load) / (1024**2)
    else:
        gpu_load_peak_mb = 0

    client = MinerUClient(
        backend="vllm-engine",
        vllm_llm=llm
    )

    print(f"✅ MinerU inicializado en {load_time:.2f}s, memoria usada durante la carga: {gpu_load_peak_mb:.2f} MB")

# -----------------------------
# 🔹 Shutdown Event
# -----------------------------
@app.on_event("shutdown")
async def shutdown_event():
    global llm, client

    if llm:
        llm.shutdown()
        llm = None
        print("🛑 LLM MinerU cerrado")

    client = None

# -----------------------------
# 🔹 Endpoint Benchmark
# -----------------------------
@app.post("/benchmark/")
async def benchmark_image(file: UploadFile = File(...)):
    global client, llm, gpu_before_load, load_time

    if client is None or llm is None:
        return JSONResponse({"error": "Modelo no inicializado"}, status_code=500)

    try:
        # Leer imagen
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Inicializar medición de GPU durante inferencia
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_init, total_init = torch.cuda.mem_get_info()
            used_peak = total_init - free_init
        else:
            used_peak = 0

        running = True

        # Hilo para monitorear GPU en tiempo real
        def monitor_gpu():
            nonlocal used_peak
            while running:
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    used = total - free
                    if used > used_peak:
                        used_peak = used
                time.sleep(0.01)  # cada 10ms

        monitor_thread = threading.Thread(target=monitor_gpu)
        monitor_thread.start()

        # Medir tiempo de inferencia
        start_infer = time.time()
        _ = client.two_step_extract(image)
        inference_time = time.time() - start_infer

        # Detener monitoreo
        running = False
        monitor_thread.join()

        # Consumo neto de GPU durante inferencia sobre la carga
        gpu_inference_peak_mb = max((used_peak - gpu_before_load) / (1024**2), 0)

        return {
            "load_time_s": round(load_time, 3),
            "gpu_load_peak_mb": round(gpu_load_peak_mb, 2),
            "inference_time_s": round(inference_time, 3),
            "gpu_inference_peak_mb": round(gpu_inference_peak_mb, 2)
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)