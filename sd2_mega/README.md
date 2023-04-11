## Run BentoML service

You can run a BentoML service with:

```
bentoml serve service:svc --production
```

Then you can test the service with `../txt2img_test.sh`. If [`xformers`](https://github.com/facebookresearch/xformers) is installed, BentoML will utilize it for inference acceleration automatically.

Alternatively, you can also run a BentoML service with web UI powered by [gradio](https://gradio.app) with:

```
bentoml serve service_with_gradio:svc --production
```

and then visit <https://127.0.0.1:3000/ui/> to generate images using web UI.

## Build a Bento

We can easily build a Bento of our service, which can be containerized to a docker image later or deployed to cloud service like AWS EC2.

To build a bento, we run `bentoml build`. Then after build we can list bentos by running `bentoml list`, the outputs may looks like:

```
 Tag                                        Size        Creation Time        Path
 anything_v3:5tts2vvn2oaggasc               18.24 GiB   2023-02-16 16:29:37  ~/bentoml/bentos/anything_v3/5tts2vvn2oaggasc
 stable_diffusion_v2_mega:mlb3s6vny6hsuasc       19.33 GiB   2023-02-16 14:59:54  ~/bentoml/bentos/stable_diffusion_v2_mega/mlb3s6vny6hsuasc
 ```

## Containerizing to docker image

You can create a docker image containing this diffusers service with all its dependencies and cuda/cudnn runtime by running `bentoml containerize stable_diffusion_v2_mega:mlb3s6vny6hsuasc`. The resulting docker image can be run on any machine with docker and `nvidia-docker` installed (`nvidia-docker` is only required if you want to run the model on GPU). This docker image bundles CUDA version 11.7, which also requires host machine has nvidia-driver version >= 515. You can modify the bundled CUDA version by changing `bentofile.yaml`.

## Deploy to AWS EC2

Please follow section "Deploy The Stable Diffusion Bento To EC2" in [this blog post](https://modelserving.com/blog/deploying-your-own-stable-diffusion-service-mz9wk)
