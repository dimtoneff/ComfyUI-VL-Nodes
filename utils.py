
import os
import numpy as np
from PIL import Image
import logging
import folder_paths
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

try:
    # For Pillow >= 9.1.0, Resampling is available
    from PIL.Image import Resampling as PILResampling

    class Resampling:
        LANCZOS = PILResampling.LANCZOS
except ImportError:
    # For older Pillow versions, fallback to Image.LANCZOS
    class Resampling:
        LANCZOS = Image.LANCZOS


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


any_type = AlwaysEqualProxy("*")


def tensor2pil(image):
    if image.dim() == 4:
        image = image.squeeze(0)
    image = image.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def resize_pil_image(pil_image, target_size=512, node_name="Node"):
    """Resizes a PIL image if its largest dimension exceeds the target size, maintaining aspect ratio."""
    w, h = pil_image.size
    if max(w, h) > target_size:
        if w > h:
            new_w = target_size
            new_h = int(h * (target_size / w))
        else:
            new_h = target_size
            new_w = int(w * (target_size / h))
        print(
            f"{node_name}: Resizing image from {w}x{h} to {new_w}x{new_h}")
        pil_image = pil_image.resize(
            (new_w, new_h), Resampling.LANCZOS)
    return pil_image


def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next(
        (x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(
            f"Unknown file list already present on key {key}: {base}")


def find_local_unet_models(name: str):
    """Find local UNet models matching the given name, searching recursively."""
    try:
        # Use a dictionary to maintain the mapping between model name and path
        local_model_map = {}
        # Scan all unet folders for local models
        for base_dir in folder_paths.get_folder_paths("unet"):
            if os.path.isdir(base_dir):
                # Use os.walk to recursively search for model directories
                for dirpath, dirnames, filenames in os.walk(base_dir, followlinks=True):
                    # A directory containing 'config.json' is considered a model directory.
                    if "config.json" in filenames:
                        model_name = os.path.basename(dirpath)
                        if name.lower() in model_name.lower() and model_name not in local_model_map:
                            local_model_map[model_name] = dirpath
                        # Don't traverse further into a model's subdirectories (e.g., snapshots)
                        dirnames[:] = []

        # Sort by model name for consistent ordering and unpack into two lists
        sorted_items = sorted(local_model_map.items())
        local_models = [item[0]
                        for item in sorted_items] if sorted_items else []
        local_models_paths = [item[1]
                              for item in sorted_items] if sorted_items else []
    except Exception as e:
        print(
            f"Unet-Loader: Could not scan for local models named {name} in unet folder: {e}")
        local_models = []
        local_models_paths = []
    return local_models, local_models_paths


class CustomQwen25VLChatHandler(Qwen25VLChatHandler):
    """
    Custom Chat Handler that inherits from Qwen25VLChatHandler and adds
    proper resource management (__del__ and close) to prevent VRAM leaks.
    """

    def close(self):
        """Explicitly close the handler and release C-level resources."""
        # The parent Llava15ChatHandler creates the _exit_stack.
        if hasattr(self, '_exit_stack'):
            self._exit_stack.close()

    def __del__(self):
        """Ensure resources are released upon garbage collection."""
        self.close()
