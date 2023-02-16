import torch

import bentoml
from bentoml.io import Image, JSON, Multipart

bento_model = bentoml.diffusers.get("sd2:latest")
stable_diffusion_runner = bento_model.to_runner()

svc = bentoml.Service("stable_diffusion_v2", runners=[stable_diffusion_runner])

@svc.api(input=JSON(), output=Image())
def txt2img(input_data):
    images, _ = stable_diffusion_runner.run(**input_data)
    return images[0]
