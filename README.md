## LEdits++
LEdits++ is a textual image editing method for Stable Diffusion XL and variants. See the [paper](https://arxiv.org/abs/2301.12247), Hugging Face [demo](https://huggingface.co/spaces/editing-images/leditsplusplus) and [docs](https://huggingface.co/docs/diffusers/v0.27.2/en/api/pipelines/ledits_pp) for details.

## How to use the API
You need to have Cog and Docker installed to run this model locally. To build the docker image with cog and run a prediction:

```
cog predict -i image=@examples/tennis.jpg -i editing_prompts="tennis ball, tomato" -i skip=0.20 -i reverse_editing_directions="True, False" -i edit_guidance_scale="5.0, 10.0" -i edit_threshold="0.9, 0.85"
```

To start a server and send requests to your locally or remotely deployed API:

```
cog run -p 5000 python -m cog.server.http
```

To edit an image with LEdits++, upload an image and specify the objects you would like to remove or add as a comma separated string. In order to edit an object in place (e.g. changing an apple to an orange), both nouns need to included in the *editing_prompts* as removal and addition targets respectively. The full list of API arguments are as follows: 

- **image:** Input image to edit.  
- **num_inversion_steps:** Number of image inversion steps to retrieve the image latent code.
- **negative_prompt:** Negative prompt for the first text encoder to guide the image generation. *optional*, defaults to None.   
- **source_prompt:** Prompt describing the input image that will be used for guidance during inversion. Guidance is disabled if the `source_prompt` is `""`.  
- **source_guidance_scale:** Strength of guidance during inversion.
- **skip:** Portion of initial steps that will be ignored for inversion and subsequent generation. Lower values will lead to stronger changes to the input image.  
- **negative_prompt2:** Negative prompt for the second text encoder to guide the image generation. *optional*, defaults to None if *negative_prompt* is also left empty, alternatively defaults to *negative_prompt* otherwise.  
- **editing_prompts:** Comma separated objects to add, remove or edit. Defaults to None, which inverts and reconstructs the input image.  
- **reverse_editing_directions:** Comma separated True or False boolean values indicating whether the corresponding prompt in `editing_prompts` should be increased or decreased to add, remove or edit. *optional*, defaults to `False`.  
- **edit_guidance_scale:** Comma separated float values for each change specified in editing prompts list. *optional*, defaults to 5 if left empty.  
- **edit_warmup_steps:** Number of diffusion steps (for each prompt) for which guidance is not applied.  
- **edit_threshold:** Comma separated edit threshold float values for each editing prompt, threshold values should be proportional to the image region that is modified. *optional*, defaults to 0.9 if left empty.  

LEdits++ only supports *DPMSolverMultistepScheduler* (default) and *DDIMScheduler*. Images that are larger than 1024x1024 are downsampled while preserving the aspect ratio.

## References
```
@article{Brack2023LEDITSLI,
  title={LEDITS++: Limitless Image Editing using Text-to-Image Models},
  author={Manuel Brack and Felix Friedrich and Katharina Kornmeier and Linoy Tsaban and Patrick Schramowski and Kristian Kersting and Apolin'ario Passos},
  journal={ArXiv},
  year={2023},
  volume={abs/2311.16711},
  url={https://api.semanticscholar.org/CorpusID:265466786}
}
```
