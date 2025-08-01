# -----------------------------------------------------------
# AUTHOR --------> Francisco Contreras
# OFFICE --------> Senior VFX Compositor, Software Developer
# WEBSITE -------> https://vinavfx.com
# -----------------------------------------------------------
import nodes
import torch
import comfy.utils
import comfy.model_management
import node_helpers
import comfy.latent_formats
from nodes import MAX_RESOLUTION
import torch.nn.functional as F


class basic_node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hola Mundo!"})
            }
        }

    CATEGORY = 'vina'
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"

    def process(self, text):
        return (text + " desde mi nodo!",)


class wan22_inpainting:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE", ),
                             "start_image": ("IMAGE", ),
                             "width": ("INT", {"default": 1280, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "height": ("INT", {"default": 704, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "length": ("INT", {"default": 49, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             }, }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = 'vina'

    def encode(self, vae, width, height, length, batch_size, start_image):
        latent = torch.zeros([1, 48, ((length - 1) // 4) + 1, height // 16,
                             width // 16], device=comfy.model_management.intermediate_device())
        mask = torch.ones([latent.shape[0], 1, ((length - 1) // 4) + 1, latent.shape[-2],
                          latent.shape[-1]], device=comfy.model_management.intermediate_device())

        start_image = comfy.utils.common_upscale(
            start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        latent_temp = vae.encode(start_image)
        latent[:, :, :latent_temp.shape[-3]] = latent_temp
        mask[:, :, :latent_temp.shape[-3]] *= 0.0

        out_latent = {}
        latent_format = comfy.latent_formats.Wan22()
        latent = latent_format.process_out(
            latent) * mask + latent * (1.0 - mask)
        out_latent["samples"] = latent.repeat(
            (batch_size, ) + (1,) * (latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat(
            (batch_size, ) + (1,) * (mask.ndim - 1))
        return (out_latent,)
