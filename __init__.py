# -----------------------------------------------------------
# AUTHOR --------> Francisco Contreras
# OFFICE --------> Senior VFX Compositor, Software Developer
# WEBSITE -------> https://vinavfx.com
# -----------------------------------------------------------
from .wan import wan_image_to_video

NODE_CLASS_MAPPINGS = {
    'wan_image_to_video': wan_image_to_video
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'wan_image_to_video': 'WANImageToVideo'
}
