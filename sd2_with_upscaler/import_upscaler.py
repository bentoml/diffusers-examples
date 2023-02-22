import bentoml

model_id = "stabilityai/stable-diffusion-x4-upscaler"

bentoml.diffusers.import_model(
    "sd2-upscaler",
    model_id,
)
