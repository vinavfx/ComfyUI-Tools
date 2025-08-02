# -----------------------------------------------------------
# AUTHOR --------> Francisco Contreras
# OFFICE --------> Senior VFX Compositor, Software Developer
# WEBSITE -------> https://vinavfx.com
# -----------------------------------------------------------
import nodes
import torch
import comfy.utils
import comfy.model_management
import comfy.latent_formats


class wan22_video2video:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE", ),
                             "video_image": ("IMAGE", ),
                             "start_image": ("IMAGE", ),
                             "width": ("INT", {"default": 1280, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "height": ("INT", {"default": 704, "min": 32, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "length": ("INT", {"default": 49, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4})
                             }, }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = 'vina'

    def encode(self, vae, video_image, start_image, width, height, length):
        frames = ((length - 1) // 4) + 1

        latent = torch.zeros([1, 48, frames, height // 16, width // 16], device=comfy.model_management.intermediate_device())
        mask = torch.ones([latent.shape[0], 1, frames, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())

        video = comfy.utils.common_upscale(video_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        latent_temp_video = vae.encode(video)
        latent[:, :, :frames] = latent_temp_video
        mask[:, :, :frames] = 1

        start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        latent_temp_start = vae.encode(start_image)
        n_start = latent_temp_start.shape[-3]
        latent[:, :, :n_start] = latent_temp_start
        mask[:, :, :n_start] = 0

        out_latent = {}
        out_latent["samples"] = latent
        out_latent["noise_mask"] = mask
        return (out_latent,)
