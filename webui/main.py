# path: webui/main.py
import base64
import httpx
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Face Recognition WebUI")
templates = Jinja2Templates(directory="webui/templates")

API_URL = "http://localhost:8000/api/v1/identify"  # رابط API الأساسي

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "image_data": None})

@app.post("/identify", response_class=HTMLResponse)
async def identify(request: Request, image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()

        # call API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                API_URL,
                files={"image": (image.filename, image_bytes, image.content_type)},
            )
        result = response.json()

        if "matches" in result and result["matches"]:
            person_name = result["matches"][0].get("name") or result["matches"][0]["person_id"]
        else:
            person_name = "Unknown"

        # نحول الصورة لـ base64 عشان نعرضها في الـ HTML
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = image.content_type

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": person_name, "image_data": f"data:{mime_type};base64,{image_data}"},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": f"Error: {str(e)}", "image_data": None},
        )
