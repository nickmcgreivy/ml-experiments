from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@app.get("/")
def serve_frontend():
    return FileResponse("frontend.html")