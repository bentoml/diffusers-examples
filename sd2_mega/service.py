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

@svc.api(input=JSON(), output=Image())
def txt2img(input_data):
    images, _ = stable_diffusion_runner.text2img.run(**input_data)
    return images[0]

img2img_input_spec = Multipart(img=Image(), data=JSON())
@svc.api(input=img2img_input_spec, output=Image())
def img2img(img, data):
    data["image"] = img
    images, _ = stable_diffusion_runner.img2img.run(**data)
    return images[0]
