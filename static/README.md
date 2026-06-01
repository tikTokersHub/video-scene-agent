# Video Scene Agent Frontend

Static landing page split into separate files.

```text
video-scene-agent-frontend/
  index.html
  css/styles.css
  js/main.js
  assets/
    images/
```

## FastAPI serving example

Place this folder in your project root as `static/`, or copy the files into your existing static directory.

```python
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"

app.mount("/css", StaticFiles(directory=STATIC_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=STATIC_DIR / "js"), name="js")
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

@app.get("/")
def landing_page():
    return FileResponse(STATIC_DIR / "index.html")
```

The Ask box calls `POST /ask` with:

```json
{ "question": "Was anyone walking on the grass?" }
```

It expects an `AgentAnswer`-style response containing `answer`, `classification`, `confidence`, and `evidence`.
