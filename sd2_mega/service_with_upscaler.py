import typing as t

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline

import bentoml
from PIL.Image import Image

sample_txt2img_input = dict(
    prompt="photo a majestic sunrise in the mountains, best quality, 4k",
    negative_prompt="blurry, low-res, ugly, low quality",
    height=320,
    width=320,
    num_inference_steps=50,
    guidance_scale=7.5,
    eta=0.0,
    upscale=True
)

sample_img2img_input = dict(
    prompt="make the image black and white", 
    strength=0.8,
    upscale=True
)

@bentoml.service(
    resources={"memory": "500MiB"},
    traffic={"timeout": 60},
)
class StableDiffusionWithUpscaler:
    sd2_model = bentoml.models.get("sd2:latest")
    upscaler_model = bentoml.models.get("sd2-upscaler:latest")

    def __init__(self) -> None:
        # Load model into pipeline
        self.stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(self.sd2_model.path, use_safetensors=True)
        self.stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
            vae=self.stable_diffusion_txt2img.vae,
            text_encoder=self.stable_diffusion_txt2img.text_encoder,
            tokenizer=self.stable_diffusion_txt2img.tokenizer,
            unet=self.stable_diffusion_txt2img.unet,
            scheduler=self.stable_diffusion_txt2img.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.upscaler_model_pipeline = StableDiffusionUpscalePipeline.from_pretrained(self.upscaler_model.path, use_safetensors=True)
        self.stable_diffusion_txt2img.to('cuda')
        self.stable_diffusion_img2img.to('cuda')
        self.upscaler_model_pipeline.to('cuda')

    @bentoml.api
    def txt2img(self, input_data: t.Dict[str, t.Any] = sample_txt2img_input) -> Image:
        upscale = input_data.pop("upscale")
        res = self.stable_diffusion_txt2img(**input_data)
        images = res[0]
        if upscale:
            prompt = input_data["prompt"]
            negative_prompt = input_data.get("negative_prompt")
            low_res_img = images[0]
            low_res_img.format = "PNG"
            res = self.upscaler_model_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=low_res_img
            )
            images = res[0]
        return images[0]

    @bentoml.api
    def img2img(self, image: Image, input_data: t.Dict[str, t.Any] = sample_img2img_input) -> Image:
        upscale = input_data.pop("upscale")
        input_data["image"] = image
        res = self.stable_diffusion_img2img(**input_data)
        images = res[0]
        if upscale:
            prompt = input_data["prompt"]
            negative_prompt = input_data.get("negative_prompt")
            low_res_img = images[0]
            low_res_img.format = "PNG"
            res = self.upscaler_model_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=low_res_img
            )
            images = res[0]
        return images[0]
    