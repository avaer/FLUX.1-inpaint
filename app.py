from typing import Tuple

# import requests
import random
import numpy as np
# import gradio as gr
# import spaces
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline

# MARKDOWN = """
# # FLUX.1 Inpainting 🔥

# Shoutout to [Black Forest Labs](https://huggingface.co/black-forest-labs) team for 
# creating this amazing model, and a big thanks to [Gothos](https://github.com/Gothos) 
# for taking it to the next level by enabling inpainting with the FLUX.
# """

MAX_SEED = np.iinfo(np.int32).max
IMAGE_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# def remove_background(image: Image.Image, threshold: int = 50) -> Image.Image:
#     image = image.convert("RGBA")
#     data = image.getdata()
#     new_data = []
#     for item in data:
#         avg = sum(item[:3]) / 3
#         if avg < threshold:
#             new_data.append((0, 0, 0, 0))
#         else:
#             new_data.append(item)

#     image.putdata(new_data)
#     return image


# EXAMPLES = [
#     [
#         {
#             "background": Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-image.png", stream=True).raw),
#             "layers": [remove_background(Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-mask-2.png", stream=True).raw))],
#             "composite": Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-composite-2.png", stream=True).raw),
#         },
#         "little lion",
#         42,
#         False,
#         0.85,
#         30
#     ],
#     [
#         {
#             "background": Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-image.png", stream=True).raw),
#             "layers": [remove_background(Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-mask-3.png", stream=True).raw))],
#             "composite": Image.open(requests.get("https://media.roboflow.com/spaces/doge-2-composite-3.png", stream=True).raw),
#         },
#         "tribal tattoos",
#         42,
#         False,
#         0.85,
#         30
#     ]
# ]

pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(DEVICE)


def resize_image_dimensions(
    original_resolution_wh: Tuple[int, int],
    maximum_dimension: int = IMAGE_SIZE
) -> Tuple[int, int]:
    width, height = original_resolution_wh

    # if width <= maximum_dimension and height <= maximum_dimension:
    #     width = width - (width % 32)
    #     height = height - (height % 32)
    #     return width, height

    if width > height:
        scaling_factor = maximum_dimension / width
    else:
        scaling_factor = maximum_dimension / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    new_width = new_width - (new_width % 32)
    new_height = new_height - (new_height % 32)

    return new_width, new_height


# @spaces.GPU(duration=100)
# def process(
#     input_image_editor: dict,
#     input_text: str,
# ):
#     input_text = ""

#     image = input_image_editor['background']
#     mask = input_image_editor['layers'][0]

#     width, height = resize_image_dimensions(original_resolution_wh=image.size)
#     resized_image = image.resize((width, height), Image.LANCZOS)
#     resized_mask = mask.resize((width, height), Image.LANCZOS)

#     seed_slicer = 0
#     randomize_seed_checkbox = True
#     strength_slider = 0.85
#     num_inference_steps_slider = 20

#     if randomize_seed_checkbox:
#       seed_slicer = random.randint(0, MAX_SEED)
#     generator = torch.Generator().manual_seed(seed_slicer)
#     result = pipe(
#         prompt=input_text,
#         image=resized_image,
#         mask_image=resized_mask,
#         width=width,
#         height=height,
#         strength=strength_slider,
#         generator=generator,
#         num_inference_steps=num_inference_steps_slider
#     ).images[0]
#     print('INFERENCE DONE')
#     return result, resized_mask

# import cv2
import io
# import torch
import PIL
# import numpy as np
# from torchvision.ops import box_convert

# from typing import Tuple, List
# import json
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# res = await fetch(`http://127.0.0.1:1234/images/image (967).png`); b = await res.blob(); fd = new FormData(); fd.append('image', b); fd.append('query', JSON.stringify(['plant'])); res = await fetch('/dino', { method: 'POST', body: fd, }); j = await res.json();

@app.route('/flux-inpaint', methods=['POST'])
def dino():
    try:
        # Get the options from FormData
        prompt = request.form.get('prompt')
        strength_string = request.form.get('strength')
        num_inference_steps_string = request.form.get('num_inference_steps')
        quality_string = request.form.get('quality')
        lossless_string = request.form.get('lossless')
        # open image data with PIL
        image_data = request.files.get('image')
        image = PIL.Image.open(io.BytesIO(image_data.read()))
        mask_data = request.files.get('mask')
        mask = PIL.Image.open(io.BytesIO(mask_data.read()))
        
        print("prompt:")
        print(prompt)
        print("image:")
        print(image)
        print("mask:")
        print(mask)
        
        width, height = resize_image_dimensions(original_resolution_wh=image.size)
        resized_image = image.resize((width, height), Image.LANCZOS)
        resized_mask = mask.resize((width, height), Image.LANCZOS)

        seed_slicer = 0
        randomize_seed_checkbox = True
        if randomize_seed_checkbox:
            seed_slicer = random.randint(0, MAX_SEED)
        strength_slider = float(0.85)
        if strength_string:
            strength_slider = float(strength_string)
        num_inference_steps_slider = int(20)
        if num_inference_steps_string:
            num_inference_steps_slider = int(num_inference_steps_string)
        quality = int(75)
        if quality_string:
            quality = int(quality_string)
        lossless = False
        if lossless_string:
            lossless = bool(lossless)
        
        generator = torch.Generator().manual_seed(seed_slicer)
        result = pipe(
            prompt=prompt,
            image=resized_image,
            mask_image=resized_mask,
            width=width,
            height=height,
            strength=strength_slider,
            generator=generator,
            num_inference_steps=num_inference_steps_slider
        ).images[0]
        print('INFERENCE DONE')
        
        # encode the result as webp
        img_byte_array = io.BytesIO()
        result.save(img_byte_array, format='WEBP', quality=quality, lossless=lossless)
        response_bytes = img_byte_array.getvalue()

        response = Response(response_bytes, content_type='image/webp')
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        
        return response
    except Exception as e:
        # Return error message and set status code to 500
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11111, use_reloader=False)