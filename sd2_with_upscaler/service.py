import typing as t

from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline

import bentoml
from PIL.Image import Image

sample_txt2img_input = dict(
    prompt="photo of a majestic sunrise in the mountains, best quality, 4k",
    negative_prompt="blurry, low-res, ugly, low quality",
    height=320,
    width=320,
    num_inference_steps=50,
    guidance_scale=7.5,
    eta=0.0
)

@bentoml.service(
    resources={"memory": "500MiB"},
    traffic={"timeout": 60},
)
class StableDiffusionUpscaler:
    sd2_model = bentoml.models.get("sd2:latest")
    upscaler_model = bentoml.models.get("sd2-upscaler:latest")

    def __init__(self) -> None:
        # Load model into pipeline
        self.sd2_model_pipeline = StableDiffusionPipeline.from_pretrained(self.sd2_model.path, use_safetensors=True)
        self.upscaler_model_pipeline = StableDiffusionUpscalePipeline.from_pretrained(self.upscaler_model.path, use_safetensors=True)
        self.sd2_model_pipeline.to('cuda')
        self.upscaler_model_pipeline.to('cuda')

    @bentoml.api
    def txt2img(self, input_data: t.Dict[str, t.Any] = sample_txt2img_input) -> Image:
        prompt = input_data["prompt"]
        negative_prompt = input_data.get("negative_prompt")
        low_res_img = self.sd2_model_pipeline(**input_data)[0][0]
        image = self.upscaler_model_pipeline(prompt=prompt, negative_prompt=negative_prompt, image=low_res_img)[0][0]
        return image
