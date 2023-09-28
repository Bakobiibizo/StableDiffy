import os
import torch
from torch.cuda.amp.autocast_mode import autocast
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


def run_stable_diffusion(prompt: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "D:/03.Models/models/checkpoints/stable-diffusion/illuminatiDiffusionV1_v11-unclip-h-fp16.safetensors"

    auth_token = os.getenv("HUGGINGFACE_TOKEN")

    pipeline = StableDiffusionPipeline.from_single_file(
        model_id,
        revision="fp16",
        use_auth_token=auth_token,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipeline.to(device)
    image = pipeline(prompt=prompt, guidance_scale=7.5, num_inference_steps=50).images[
        0
    ]
    count = len(os.listdir("E:/01Coding/stable-diffy-react/output"))
    image.save(f"E:/01Coding/stable-diffy-react/output/output{count+1}.png")
    return image
