from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.api.routes import router

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="LLM Fine-Tuning Engine",
    description="QLoRA/LoRA fine-tuning UI — configs, datasets, train, export GGUF, chat",
    version="0.1.0",
)

app.include_router(router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def main():
    import os
    import sys
    import uvicorn

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    uvicorn.run("web.app:app", host="127.0.0.1", port=7860, reload=False)


if __name__ == "__main__":
    main()
