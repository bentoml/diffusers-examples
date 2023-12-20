This is a BentoML service that integrate Stable Diffusion 2.0 with the new [Stable Diffusion x4 upscaler model](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler). The upscaler model is very memory consuming. It can barely upscale a 512x512 image on a Nvidia GPU with 24GB VRAM when [`xformers`](https://github.com/facebookresearch/xformers) is installed.

## Import upscaler model

```
python3 import_upscaler.py
```

## Run BentoML service

You can run a BentoML service with:

```
bentoml serve service:StableDiffusionUpscaler --production
```

Then you can test the service with `../txt2img_test.sh`. If [`xformers`](https://github.com/facebookresearch/xformers) is installed, BentoML will utilize it for inference acceleration automatically.

## Build a Bento

We can easily build a Bento of our service, which can be containerized to a docker image later or deployed to cloud service like AWS EC2.

To build a bento, we run `bentoml build`. Then after build we can list bentos by running `bentoml list`, the outputs may looks like:

```
 Tag                                   Size       Creation Time        Path
 stable_diffusion_v2:r325zpfsekpm74r4  32.38 GiB  2023-02-21 20:02:48  ~/bentoml/bentos/stable_diffusion_v2/r325zpfsekpm74r4
 stable_diffusion_v2:axaci4vsecdv34r4  32.38 GiB  2023-02-21 19:44:33  ~/bentoml/bentos/stable_diffusion_v2/axaci4vsecdv34r4
```

## Containerizing to docker image

You can create a docker image containing this diffusers service with all its dependencies and cuda/cudnn runtime by running `stable_diffusion_v2:axaci4vsecdv34r4`. The resulting docker image can be run on any machine with docker and `nvidia-docker` installed (`nvidia-docker` is only required if you want to run the model on GPU). This docker image bundles CUDA version 11.7, which also requires host machine has nvidia-driver version >= 515. You can modify the bundled CUDA version by changing `bentofile.yaml`.

## Deploy to AWS EC2

Please follow section "Deploy The Stable Diffusion Bento To EC2" in [this blog post](https://modelserving.com/blog/deploying-your-own-stable-diffusion-service-mz9wk)
