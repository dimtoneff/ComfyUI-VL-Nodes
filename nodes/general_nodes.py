import time
import os
import gc
import torch
import requests
from comfy.cli_args import args
import comfy.model_management
import comfy.utils
from PIL import Image, ImageOps
import numpy as np
from .mimo_nodes import unload_all_mimo_models
from .lfm2_nodes import unload_all_lfm2_hf_models
from .ovisu1_nodes import unload_all_ovisu1_models
from .ovis25_nodes import unload_all_ovis25_models
from .keye_nodes import unload_all_keye_models
from ..utils import any_type, sort_by

sort_methods = [
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)"
]


class LoadImagesFromDirBatch:
    """
    Load a batch of images from a directory. Inspired from InspirePack with some modifications.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (sort_methods,),
                "filename_text_extension": (["true", "false"], {"default": "false"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "COUNT", "FILE_NAME")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "load_images"

    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_method=None, filename_text_extension="false"):
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory '{directory} cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(
            f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sort_by(dir_files, directory, sort_method)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        file_paths = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        has_non_empty_mask = False

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                has_non_empty_mask = True
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)

            filename = os.path.basename(image_path)
            if filename_text_extension == "false":
                filename = os.path.splitext(filename)[0]
            file_paths.append(filename)

            image_count += 1

        if len(images) == 0:
            return (None, None, 0, tuple())

        if len(images) == 1:
            return (images[0], masks[0], 1, tuple(file_paths))

        elif len(images) > 1:
            image1 = images[0]
            mask1 = None

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(
                        image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(
                            image1.shape[1], image1.shape[2]), mode='bilinear', align_corners=False)
                        mask2 = mask2.squeeze(0)
                    else:
                        mask2 = mask2.unsqueeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)

                if mask1 is None:
                    mask1 = mask2
                else:
                    mask1 = torch.cat((mask1, mask2), dim=0)

            return (image1, mask1, len(images), tuple(file_paths))


class LoadImagesFromDirList:
    """
    Load a batch of images from a directory. Inspired from InspirePack with some modifications.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (sort_methods,),
                "filename_text_extension": (["true", "false"], {"default": "false"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FILE_NAME")
    OUTPUT_IS_LIST = (True, True, True)

    FUNCTION = "load_images"

    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_method=None, filename_text_extension="false"):
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Directory '{directory}' cannot be found.'")
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']

        dir_files = [f for f in dir_files if any(
            f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sort_by(dir_files, directory, sort_method)
        dir_files = [os.path.join(directory, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        file_paths = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            images.append(image)
            masks.append(mask)

            filename = os.path.basename(image_path)
            if filename_text_extension == "false":
                filename = os.path.splitext(filename)[0]
            file_paths.append(filename)

            image_count += 1

        return (images, masks, file_paths)


class MiMoFreeMemoryAPI:
    def __init__(self):
        self.add_waiting = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"anything": (any_type, {})},
            "optional": {"add_waiting": ("BOOLEAN", {"default": False, "label_on": "Enable Waiting", "label_off": "Disable Waiting"})},
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "free_memory"
    CATEGORY = "VL-Nodes"
    OUTPUT_NODE = True

    def unload_models(self):
        print("MiMoUnloadModels: Clearing all models from memory.")

        # Unload MiMo models first, as they have special handling
        unload_all_mimo_models()

        # Unload LFM2 HF models
        unload_all_lfm2_hf_models()

        # Unload Ovis-U1 models
        unload_all_ovisu1_models()

        # Unload Ovis-2.5 models
        unload_all_ovis25_models()

        # Unload Keye models
        unload_all_keye_models()

        if self.add_waiting:
            time.sleep(1)

        # Then call Comfy's global unload function for all other models (SD, VAE, etc.)
        comfy.model_management.unload_all_models()
        if self.add_waiting:
            time.sleep(1)

        # A final garbage collect to be sure everything is cleaned up.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.add_waiting:
            time.sleep(1)

    def free_memory(self, anything, add_waiting=False):
        self.add_waiting = add_waiting  # Store the value for use in unload_models
        self.unload_models()
        if self.add_waiting:
            time.sleep(1)
        host = args.listen if args.listen != "0.0.0.0" else "127.0.0.1"
        port = args.port
        url = f"http://{host}:{port}/free"

        payload = {"unload_models": True, "free_memory": True}

        try:
            print(
                f"FreeMemoryAPI: Calling ComfyUI API to free memory: {url}")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print("FreeMemoryAPI: Successfully freed models and execution cache.")
            if self.add_waiting:
                time.sleep(1)
        except Exception as e:
            print(f"FreeMemoryAPI: Failed to call /free API: {e}")

        return (anything,)


NODE_CLASS_MAPPINGS = {
    "MiMoFreeMemoryAPI": MiMoFreeMemoryAPI,
    "LoadImagesFromDirBatch": LoadImagesFromDirBatch,
    "LoadImagesFromDirList": LoadImagesFromDirList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiMoFreeMemoryAPI": "Free Memory (VL Nodes)",
    "LoadImagesFromDirBatch": "Load Images/FileNames from Dir (Batch)(VL)",
    "LoadImagesFromDirList": "Load Images/FileNames from Dir (List)(VL)",
}
