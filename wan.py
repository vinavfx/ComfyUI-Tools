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
                             "video_image": ("IMAGE", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                             "width": ("INT", {"default": 1280, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "height": ("INT", {"default": 704, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "length": ("INT", {"default": 49, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             }, }


    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = 'vina'

    def encode(self, vae, video_image, start_image, end_image, width, height, length, batch_size):

        # se dividen por 4 para que sea mas liviano
        frames = ((length - 1) // 4) + 1

        latent = torch.zeros([1, 48, frames, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        mask = torch.ones([latent.shape[0], 1, frames, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())

        video = comfy.utils.common_upscale(video_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        latent_temp_video = vae.encode(video)
        video_frames = latent_temp_video.shape[-3]
        latent[:, :, :frames] = latent_temp_video
        mask[:, :, :frames] = 1

        #  start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        #  latent_temp_start = vae.encode(start_image)
        #  n_start = latent_temp_start.shape[-3]
        #  latent[:, :, :n_start] = latent_temp_start
        #  mask[:, :, :n_start] = 0.0

        #  end_image = comfy.utils.common_upscale(end_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        #  latent_temp_end = vae.encode(end_image)
        #  n_end = latent_temp_end.shape[-3]
        #  latent[:, :, -n_end:] = latent_temp_end
        #  mask[:, :, -n_end:] = 0.0


        out_latent = {}
        latent_format = comfy.latent_formats.Wan22()
        latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)
        out_latent["samples"] = latent.repeat((batch_size, ) + (1,) * (latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size, ) + (1,) * (mask.ndim - 1))
        return (out_latent,)

