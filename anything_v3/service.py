import torch

import bentoml
from bentoml.io import Image, JSON, Multipart

bento_model = bentoml.diffusers.get("anything-v3:latest")
anything_v3_runner = bento_model.with_options(
    torch_dtype=torch.float16,
).to_runner()

svc = bentoml.Service("anything_v3", runners=[anything_v3_runner])

@svc.api(input=JSON(), output=Image())
def txt2img(input_data):
    images, _ = anything_v3_runner.run(**input_data)
    return images[0]
