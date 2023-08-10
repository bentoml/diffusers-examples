import typing as t

from pydantic import BaseModel

import torch
from diffusers import DiffusionPipeline

import bentoml
from bentoml.io import Image, JSON, Multipart

bento_model = bentoml.diffusers.get("sd2:latest")
stable_diffusion_runner = bento_model.with_options(
    pipeline_class=DiffusionPipeline,
    custom_pipeline="stable_diffusion_mega",
).to_runner()

svc = bentoml.Service("stable_diffusion_v2_mega", runners=[stable_diffusion_runner])

# text2img input validation
class Text2ImgSDArgs(BaseModel):
    prompt: str
    negative_prompt: t.Optional[str] = None
    height: t.Optional[int] = 768
    width: t.Optional[int] = 768
    num_inference_steps: t.Optional[int] = 50
    guidance_scale: t.Optional[float] = 7.5
    eta: t.Optional[float] = 0.0

text2img_input_sample = Text2ImgSDArgs(
    prompt="photo a majestic sunrise in the mountains, best quality, 4k",
    negative_prompt="blurry, low-res, ugly, low quality",
    height=768,
    width=768,
    num_inference_steps=50,
    guidance_scale=7.5,
    eta=0.0,
)
text2img_input_spec = JSON.from_sample(text2img_input_sample)

@svc.api(input=text2img_input_spec, output=Image())
def text2img(input_data):
    kwargs = input_data.dict()
    res = stable_diffusion_runner.text2img.run(**kwargs)
    images = res[0]
    return images[0]

# img2img input validation
class Img2ImgSDArgs(Text2ImgSDArgs):
    strength: t.Optional[float] = 0.8

img2img_input_spec = Multipart(img=Image(), data=JSON(pydantic_model=Img2ImgSDArgs))
@svc.api(input=img2img_input_spec, output=Image())
def img2img(img, data):
    kwargs = data.dict()
    kwargs["image"] = img
    res = stable_diffusion_runner.img2img.run(**kwargs)
    images = res[0]
    return images[0]
