import torch
import PIL
import requests
from io import BytesIO
from diffusers import LEditsPPPipelineStableDiffusionXL, AutoencoderKL

device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(
    base_model_id, vae=vae, torch_dtype=torch.float16
).to(device)


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/tennis.jpg"
image = download_image(img_url)

downsample_ratio = 1024 / image.size[0]
width = int(image.size[0] * downsample_ratio)
height = int(image.size[1] * downsample_ratio)
image = image.resize((width, height), resample=PIL.Image.LANCZOS)
# print(image)
_ = pipe.invert(image=image, num_inversion_steps=50, skip=0.2)

output = pipe(
    editing_prompt=["tennis ball", "tomato"],
    reverse_editing_direction=[True, False],
    edit_guidance_scale=[5.0, 10.0],
    edit_threshold=[0.9, 0.85],
)
output.images[0].save("output.png")
