import bentoml

bentoml.diffusers.import_model(
    "sd2",
    "stabilityai/stable-diffusion-2",
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
