"""
@author: BoundDetailers
@title: Bound Detailers
@nickname: Bound Detailers
@description: A modified detailer node that restricts feature detection and inpainting
              to a user-defined mask region. Allows per-subject control in multi-subject
              images (e.g., different eye colors for different characters).
              Requires ComfyUI-Impact-Pack.
"""

import os
import sys
import logging

# Add our module directory to the path
modules_path = os.path.dirname(os.path.abspath(__file__))
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

logging.info("[Bound Detailers] Loading Bound Detailers nodes...")

try:
    from bound_detailers_node import BoundDetailer
except ImportError as e:
    logging.error(f"[Bound Detailers] Failed to import BoundDetailer: {e}")
    logging.error("[Bound Detailers] Make sure ComfyUI-Impact-Pack is installed!")
    raise

NODE_CLASS_MAPPINGS = {
    "BoundDetailer": BoundDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoundDetailer": "Bound Detailer (Mask-Targeted)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

logging.info("[Bound Detailers] Nodes loaded successfully.")
