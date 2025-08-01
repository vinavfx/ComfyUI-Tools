# -----------------------------------------------------------
# AUTHOR --------> Francisco Contreras
# OFFICE --------> Senior VFX Compositor, Software Developer
# WEBSITE -------> https://vinavfx.com
# -----------------------------------------------------------
from .wan import wan22_inpainting


NODE_CLASS_MAPPINGS = {
    'wan22_inpainting': wan22_inpainting
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'wan22_inpainting': 'Wan22Inpainting'
}
