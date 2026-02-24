import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import router, get_model

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="Kokoro OpenAI-Compatible TTS", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router, prefix="/v1")

@app.on_event("startup")
def warm_start():
    """
    Optional: force model load on startup (reduces first-request latency).
    Set PRELOAD_ON_STARTUP=0 to disable.
    """
    preload = os.getenv("PRELOAD_ON_STARTUP", "1").strip() == "1"
    if preload:
        logging.getLogger("kokoro-api").info("Startup preload enabled: loading model...")
        get_model()
