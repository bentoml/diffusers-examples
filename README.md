# `bentoml.diffusers` examples

This repository hosts supplementary materials of {the blog post url place holder}.

Currently we have three examples:

- `sd2/` contains a service with `txt2img/` endpoint utilizing [`stabilityai/stable-diffusion-2`](https://huggingface.co/stabilityai/stable-diffusion-2)
- `sd2_mega/` contains a service with `txt2img/` and `img2img/` endpoints utilizing [`stabilityai/stable-diffusion-2`](https://huggingface.co/stabilityai/stable-diffusion-2) and diffusers' [custom pipline](https://github.com/huggingface/diffusers/tree/main/examples/community)
- `anything_v3/` contains a service with `txt2img/` endpoint utilizing [`Linaqruf/anything-v3.0`](https://huggingface.co/Linaqruf/anything-v3.0)

## Prepare environment

We recommend running these services on a machine equipped with a Nvidia graphic card and CUDA Toolkit installed.

First let's prepare a virtual environment and install requried depedencies

```
python3 -m venv venv/ && source venv/bin/activate
pip install -U -r requirements.txt
```

## Import models

You may need to authorize your huggingface account to download models, to do that, run:

```
pip install -U huggingface_hub
huggingface-cli login
```

then:

- to import [`stabilityai/stable-diffusion-2`](https://huggingface.co/stabilityai/stable-diffusion-2), run `python3 import_model.py`
- to import [`Linaqruf/anything-v3.0`](https://huggingface.co/Linaqruf/anything-v3.0), run `python3 import_anything_v3`

## Start the service and go on

After the model is imported, you can go into `sd2/`, `sd2_mega` or `anything_v3` and follow the readme inside the folder to start the service and make a docker image for each service

