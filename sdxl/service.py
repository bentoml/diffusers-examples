import typing as t

import torch
from pydantic import BaseModel

import bentoml
from bentoml.io import Image, JSON, Multipart

bento_model = bentoml.diffusers.get("sdxl-1.0:latest")
stable_diffusion_runner = bento_model.with_options(
    pipeline_class="diffusers.StableDiffusionXLPipeline",
).to_runner()

class SDArgs(BaseModel):
    prompt: str
    negative_prompt: t.Optional[str] = None
    height: t.Optional[int] = 1024
    width: t.Optional[int] = 1024
    num_inference_steps: t.Optional[int] = 50
    guidance_scale: t.Optional[float] = 7.5
    eta: t.Optional[float] = 0.0

    class Config:
        extra = "allow"

sample = SDArgs(prompt="a bento box")


svc = bentoml.Service("sdxl-service", runners=[stable_diffusion_runner])

@svc.api(input=JSON.from_sample(sample), output=Image())
def txt2img(input_data):
    kwargs = input_data.dict()
    res = stable_diffusion_runner.run(**kwargs)
    images = res[0]
    return images[0]
