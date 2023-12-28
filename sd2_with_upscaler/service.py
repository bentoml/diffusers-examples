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

sample_img2img_input = dict(
    prompt="make higher resolution", 
    strength=0.8,
    upscale=True
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
    def upscale(self,  image: Image, input_data: t.Dict[str, t.Any] = sample_img2img_input) -> Image:
        input_data["image"] = image
        prompt = input_data["prompt"]
        negative_prompt = input_data.get("negative_prompt")
        res = self.upscaler_model_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image
        )
        images = res[0]
        return images[0]
