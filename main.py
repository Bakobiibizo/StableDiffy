import json
import base64
import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from stable_diffy import run_stable_diffusion
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prompt(BaseModel):
    prompt: str
    chosen_pipeline: str


@app.get("/generate")
def generate(prompt: str):
    input_prompt = prompt
    image = run_stable_diffusion(input_prompt)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_string = base64.b64encode(buffer.getvalue())
    return Response(image_string, media_type="image/png")


class Pipeline(BaseModel):
    generation: str
    detailing: str
    upscaling: str
    post_processing: str


@app.get("/pipelines")
def pipelines(pipelines: str):
    pipelines = json.loads(pipelines)


class Styles(BaseModel):
    anime_style: str
    cyberpunk_style: str
    digital_style: str
    photoreal_style: str
    fantasy_style: str
    scifi_style: str


@app.get("/styles")
def styles(styles: str):
    styles = json.loads(styles)
    return styles


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
