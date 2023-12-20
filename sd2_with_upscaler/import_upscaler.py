import bentoml
from diffusers import StableDiffusionUpscalePipeline

model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, use_safetensors=True)

with bentoml.models.create(
    name='sd2-upscaler',
) as model_ref:
    pipeline.save_pretrained(model_ref.path)
    print(f"Model saved: {model_ref}")
