import torch
from diffusers import DiffusionPipeline

import bentoml
from bentoml.io import Image, JSON, Multipart

bento_model = bentoml.diffusers.get("sd2:latest")
stable_diffusion_runner = bento_model.with_options(
    pipeline_class=DiffusionPipeline,
    custom_pipeline="stable_diffusion_mega",
).to_runner()

svc = bentoml.Service("stable_diffusion_v2_mega_with_gradio", runners=[stable_diffusion_runner])

@svc.api(input=JSON(), output=Image())
def txt2img(input_data):
    res = stable_diffusion_runner.text2img.run(**input_data)
    images = res[0]
    return images[0]

img2img_input_spec = Multipart(img=Image(), data=JSON())
@svc.api(input=img2img_input_spec, output=Image())
def img2img(img, data):
    data["image"] = img
    res = stable_diffusion_runner.img2img.run(**data)
    images = res[0]
    return images[0]


# The following codes are for gradio web UI so we import gradio related modules here

import gradio as gr, re
from PIL import Image

def inference(prompt, guidance, steps, img=None, width=512, height=512, strength=0.5, neg_prompt=""):

    if img is None:
        # text2img
        res = stable_diffusion_runner.text2img.run(
            prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            width=width,
            height=height,
        )
        images = res[0]
        return images
    else:
        ratio = min(height / img.height, width / img.width)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
        res = stable_diffusion_runner.img2img.run(
            prompt,
            image=img,
            negative_prompt=neg_prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance,
            strength=strength,
        )
        images = res[0]
        return images

css = """
.finetuned-diffusion-div div{
    display:inline-flex;
    align-items:center;
    gap:.8rem;
    font-size:1.75rem
}
.finetuned-diffusion-div div h1{
    font-weight:900;
    margin-bottom:7px
}
.finetuned-diffusion-div p{
    margin-bottom:10px;
    font-size:94%
}
a{
    text-decoration:underline
}
.tabs{
    margin-top:0;
    margin-bottom:0
}
#gallery{
    min-height:20rem
}
"""

with gr.Blocks(css=css) as demo:

    gr.HTML(
        """
            <div>
              <div>
                <h1>Diffusion Space</h1>
              </div>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column(scale=55):
          with gr.Group():
              gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery"
              ).style(grid=[2], height="auto", container=True)
          
          settings = gr.Markdown()
          error_output = gr.Markdown()

        with gr.Column(scale=45):
          with gr.Tab("Main"):
            generate = gr.Button(value="Generate", variant="secondary").style(container=False)
            with gr.Group():
              prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=3,placeholder="Enter prompt", lines=3).style(container=False)
              
              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

              with gr.Row():
                guidance = gr.Slider(label="Guidance scale", value=7, maximum=20, step=1)
                steps = gr.Slider(label="Steps", value=20, minimum=2, maximum=50, step=1)

              with gr.Row():
                width = gr.Slider(label="Width", value=768, minimum=64, maximum=1920, step=64)
                height = gr.Slider(label="Height", value=768, minimum=64, maximum=1920, step=64)

          with gr.Tab("Img2Img"):
            with gr.Group():
              image = gr.Image(label="Image", height=256, tool="editor", type="pil")
              strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)
            generate_i2i = gr.Button(value="Generate", variant="secondary").style(container=False)


    inputs = [prompt, guidance, steps, image, width, height, strength, neg_prompt]
    outputs = [gallery]
    prompt.submit(inference, inputs=inputs, outputs=outputs, show_progress=True)
    generate.click(inference, inputs=inputs, outputs=outputs, show_progress=True)
    generate_i2i.click(inference, inputs=inputs, outputs=outputs, show_progress=True)

    exp = gr.Examples([
      ["beautiful female witch"],
      ["beautiful portrait of a girl with demon horns and blonde hair, by Ilya Kuvshinov"],
      ["centered, profile picture, simple, barbie, bright, beautiful girl, teenager, blonde hair, feminine, shimmering, sparkle, girlie, pink, princesss"],
      ["cute cotton candy girl, realistic, photo real, fantasy, pastel colors, rainbow curve hair"],
      ["orange and black, head shot of a woman standing under street lights, dark theme, Frank Miller, cinema, ultra realistic, ambiance, insanely detailed and intricate, hyper realistic, 8k resolution, photorealistic, highly textured, intricate details"],
      ["beautiful fjord at sunrise"],
      ], inputs=[prompt], outputs=outputs, fn=inference, cache_examples=False, label="Prompts")

    exnp = gr.Examples([
      ["ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"],
      ["blurry, bad art, bad anatomy, blurred, text, watermark, grainy"],
      ["ugly, deformed, disfigured, malformed, blurry, mutated, extra limbs, bad anatomy, cropped, floating limbs, disconnected limbs"],
      ["blender, cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, double, blurred, disfigured, deformed, repetitive, black and white"],
      ["bad art, strange colours, sketch, lacklustre, repetitive, cropped, lowres, deformed, old, childish"],
      ["blender, text, disfigured, realistic, photo, 3d render, fused fingers, malformed"],
      ["blender, text, disfigured, realistic, photo, 3d render, grain, cropped, out of frame"],
      ["fog blurry soft tiling bad art grainy"],
      ], inputs=[neg_prompt], outputs=outputs, fn=inference, cache_examples=False, label="Negative Prompts")

    set = gr.Examples([
          [512,512],
          [768,1024],
          [1024, 768],
          [1920, 1088],
      ], inputs=[width, height], outputs=outputs, fn=inference, cache_examples=False, label="Resolutions")

    gr.HTML("""
    <div style="border-top: 1px solid #303030;">
      <br>
      <p>adpoted from WebUI made by Nitrosocke.</p>
    </div>
    """)

svc.mount_asgi_app(demo.app, path="/ui/")
