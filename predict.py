import os
import subprocess
from typing import List

import torch
import PIL
from PIL import Image
from diffusers import LEditsPPPipelineStableDiffusionXL
from cog import BasePredictor, Input, Path


MODEL_CACHE = "sdxl-cache"
MODEL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-fix-1.0.tar"


def resize_image(image):
    max_dim = max(list(image.size))
    downsample_ratio = 1024 / max_dim

    width = int(image.size[0] * downsample_ratio)
    height = int(image.size[1] * downsample_ratio)
    image = image.resize((width, height), resample=PIL.Image.LANCZOS)

    return image


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"

        if not os.path.exists(MODEL_CACHE):
            subprocess.check_call(["pget", "-x", MODEL_URL, MODEL_CACHE])

        self.pipeline = LEditsPPPipelineStableDiffusionXL.from_pretrained(
            MODEL_CACHE, torch_dtype=torch.float16
        ).to(self.device)

    def predict(
        self,
        image: Path = Input(description="Input image to edit."),
        num_inversion_steps: int = Input(
            description="Number of image inversion steps.", ge=1, le=200, default=50
        ),
        source_prompt: str = Input(
            description="Prompt describing the input image that will be used for guidance during inversion. Guidance is disabled if the `source_prompt` is `"
            "`.",
            default="",
        ),
        source_guidance_scale: float = Input(
            description="Strength of guidance during inversion.",
            ge=1,
            le=25,
            default=3.5,
        ),
        skip: float = Input(
            description="Portion of initial steps that will be ignored for inversion and subsequent generation. Lower values will lead to stronger changes to the input image.",
            ge=0,
            le=1.0,
            default=0.15,
        ),
        negative_prompt: str = Input(
            description="Negative prompt for the first text encoder to guide the image generation. *optional*, defaults to None.",
            default=None,
        ),
        negative_prompt2: str = Input(
            description="Negative prompt for the second text encoder to guide the image generation. *optional*, defaults to None if *negative_prompt* is also left empty, alternatively defaults to *negative_prompt* otherwise.",
            default=None,
        ),
        editing_prompts: str = Input(
            description="Comma separated objects to add, remove or edit. Defaults to None, which inverts and reconstructs the input image.",
            default=None,
        ),
        reverse_editing_directions: str = Input(
            description="Comma separated True or False boolean values indicating whether the corresponding prompt in `editing_prompts` should be increased or decreased to add, remove or edit. *optional*, defaults to `False`",
            default=None,
        ),
        edit_guidance_scale: str = Input(
            description="Comma separated float values for each change specified in editing prompts list. *optional*, defaults to 5 if left empty.",
            default=None,
        ),
        edit_warmup_steps: int = Input(
            description="Number of diffusion steps (for each prompt) for which guidance is not applied",
            ge=0,
            le=100,
            default=0,
        ),
        edit_threshold: str = Input(
            description="Comma separated edit threshold float values for each editing prompt, threshold values should be proportional to the image region that is modified. *optional*, defaults to 0.9 if left empty. ",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # resize image
        image = Image.open(str(image))
        image = resize_image(image)

        # inversion / retrieve image latent code
        _ = self.pipeline.invert(
            image=image,
            num_inversion_steps=num_inversion_steps,
            source_prompt=source_prompt,
            source_guidance_scale=source_guidance_scale,
            skip=skip,
        )

        if editing_prompts is not None:
            editing_prompts = [target.strip() for target in editing_prompts.split(",")]

        if reverse_editing_directions is not None:
            editing_directions = []
            for target in reverse_editing_directions.split(","):
                if target.strip().lower() == "true":
                    editing_directions.append(True)
                elif target.strip().lower() == "false":
                    editing_directions.append(False)
                else:
                    raise Exception(
                        "Invalid editing direction, please only include comma separated True or False values."
                    )

        if edit_guidance_scale is not None:
            edit_guidance_scale = [
                float(target.strip()) for target in edit_guidance_scale.split(",")
            ]

        if edit_threshold is not None:
            edit_threshold = [
                float(target.strip()) for target in edit_threshold.split(",")
            ]

        # edit image
        output = self.pipeline(
            editing_prompt=editing_prompts,
            reverse_editing_direction=editing_directions,
            edit_threshold=edit_threshold,
            edit_guidance_scale=edit_guidance_scale,
            edit_warmup_steps=edit_warmup_steps,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt2,
            width=image.size[0],
            height=image.size[1],
        )

        out_path = "output.png"
        output.images[0].save(out_path)

        return Path(out_path)
