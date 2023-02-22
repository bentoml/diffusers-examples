import torch
from diffusers import DiffusionPipeline

import bentoml
from bentoml.io import Image, JSON, Multipart

import diffusers

bento_model = bentoml.diffusers.get("sd2:latest")
stable_diffusion_runner = bento_model.to_runner()

upscaler_model = bentoml.diffusers.get("sd2-upscaler:latest")
upscaler_runner = upscaler_model.with_options(
    pipeline_class=DiffusionPipeline,
).to_runner()

svc = bentoml.Service("stable_diffusion_v2_with_upscaler", runners=[stable_diffusion_runner, upscaler_runner])

@svc.api(input=JSON(), output=Image())
def txt2img(input_data):
    prompt = input_data["prompt"]
    negative_prompt = input_data.get("negative_prompt")
    images, _ = stable_diffusion_runner.run(**input_data)
    low_res_img = images[0]
    images = upscaler_runner.run(prompt=prompt, negative_prompt=negative_prompt, image=low_res_img)
    return images[0][0]
