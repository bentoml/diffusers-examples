import bentoml
from bentoml.io import Image, JSON, Multipart
bentoml.set_serialization_strategy("LOCAL_BENTO")

bento_model = bentoml.diffusers.get("sd2:latest")
stable_diffusion_runner = bento_model.to_runner()

svc = bentoml.Service("stable_diffusion_v2", runners=[stable_diffusion_runner])

@svc.api(input=JSON(), output=Image())
def txt2img(input_data):
    res = stable_diffusion_runner.run(**input_data)
    images = res[0]
    return images[0]
