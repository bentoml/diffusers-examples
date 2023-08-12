import bentoml

bentoml.diffusers.import_model(
    "sdxl-1.0",
    "stabilityai/stable-diffusion-xl-base-1.0",
    signatures={
        "__call__": {
            "batchable": False
        },
        "text2img": {
            "batchable": False
        },
        "img2img": {
            "batchable": False
        },
        "inpaint": {
            "batchable": False
        },
    }
)
