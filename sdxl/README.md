## Prepare environment

We recommend running these services on a machine equipped with a Nvidia graphic card and CUDA Toolkit installed.

First let's prepare a virtual environment and install requried depedencies

```
python3 -m venv venv/ && source venv/bin/activate
pip install -U -r requirements.txt
```

## Import SDXL models

You may need to authorize your huggingface account to download models, to do that, run:

```
pip install -U huggingface_hub
huggingface-cli login
```

then run:

```
python3 import_model.py
```

## Run BentoML service

You can run a BentoML service with:

```
bentoml serve service:svc
```

Then you can test the service with `../txt2img_test.sh`. If [`xformers`](https://github.com/facebookresearch/xformers) is installed, BentoML will utilize it for inference acceleration automatically.

You can also visit <http://127.0.0.1:3000/> and use swagger UI to interact with the API endpoint.

## Build a Bento

We can easily build a Bento of our service, which can be containerized to a docker image later or deployed to cloud service like AWS EC2.

To build a bento, we run `bentoml build`. Then after build we can list bentos by running `bentoml list`, the outputs may looks like:

```
 Tag                                  Size        Creation Time
 sdxl-service:soyqeary3gowwasc        21.37 KiB   2023-08-12 14:29:55
```


## Containerizing to docker image

You can create a docker image containing this diffusers service with all its dependencies and cuda/cudnn runtime by running `sdxl-service:soyqeary3gowwasc`. The resulting docker image can be run on any machine with docker and `nvidia-docker` installed (`nvidia-docker` is only required if you want to run the model on GPU). This docker image bundles CUDA version 11.8, which also requires host machine has nvidia-driver version >= 515. You can modify the bundled CUDA version by changing `bentofile.yaml`.

## Deploy to AWS EC2

Please follow section "Deploy The Stable Diffusion Bento To EC2" in [this blog post](https://modelserving.com/blog/deploying-your-own-stable-diffusion-service-mz9wk)
