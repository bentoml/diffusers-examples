import bentoml
from diffusers import StableDiffusionPipeline

model_id = 'stabilityai/stable-diffusion-2'
pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

with bentoml.models.create(
    name='sd2-model',
) as model_ref:
    pipeline.save_pretrained(model_ref.path)
    print(f"Model saved: {model_ref}")
