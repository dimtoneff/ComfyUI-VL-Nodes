import gc
import os
import re
import time

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

import comfy.model_management
import comfy.utils
from comfy.cli_args import args

from ..utils import any_type, sort_by
from .gguf_nodes import unload_all_gguf_models
from .internvl35_nodes import unload_all_internvl_models
from .keye_nodes import unload_all_keye_models
from .lfm2_nodes import unload_all_lfm2_hf_models
from .ovis25_nodes import unload_all_ovis25_models
from .ovisu1_nodes import unload_all_ovisu1_models

sort_methods = ["None", "Alphabetical (ASC)", "Alphabetical (DESC)", "Numerical (ASC)", "Numerical (DESC)", "Datetime (ASC)", "Datetime (DESC)"]


class TextSave_VL:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "path": ("STRING", {"default": './ComfyUI/output/[time(%Y-%m-%d)]', "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "filename_number_padding": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
            },
            "optional": {
                "file_extension": ("STRING", {"default": ".txt"}),
                "encoding": ("STRING", {"default": "utf-8"}),
                "filename_suffix": ("STRING", {"default": ""}),
                "skip_overwrite": ("BOOLEAN", {"default": False, "tooltip": "If enabled, will skip overwriting to files that already exist. Useful when purging bad captions and re-running workflows to re-do only the missing captions."})
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_text_file"
    CATEGORY = "VL-Nodes/Batch"

    def save_text_file(self, text, path, filename_prefix='ComfyUI', filename_delimiter='_', filename_number_padding=4, file_extension='.txt', encoding='utf-8', filename_suffix='', skip_overwrite=False):
        if not os.path.exists(path):
            print(f"The path `{path}` doesn't exist! Creating it...")
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                print(f"The path `{path}` could not be created! Is there write access?\n{e}")

        if text.strip() == '':
            print("There is no text specified to save! Text is empty.")

        delimiter = filename_delimiter
        number_padding = int(filename_number_padding)
        filename = self.generate_filename(path, filename_prefix, delimiter, number_padding, file_extension, filename_suffix)
        file_path = os.path.join(path, filename)

        # Check if we should skip overwrite and file exists
        if skip_overwrite and os.path.exists(file_path):
            print(f"Skipping overwrite of existing file: {file_path}")
        else:
            self.write_text_file(file_path, text, encoding)
        return (text, {"ui": {"string": text}})

    def generate_filename(self, path, prefix, delimiter, number_padding, extension, suffix):
        if number_padding == 0:
            # If number_padding is 0, don't use a numerical suffix
            filename = f"{prefix}{suffix}{extension}"
        else:
            if delimiter:
                pattern = f"{re.escape(prefix)}{re.escape(delimiter)}(\\d{{{number_padding}}}){re.escape(suffix)}{re.escape(extension)}"
            else:
                pattern = f"{re.escape(prefix)}(\\d{{{number_padding}}}){re.escape(suffix)}{re.escape(extension)}"

            existing_counters = [
                int(re.search(pattern, filename).group(1))
                for filename in os.listdir(path)
                if re.match(pattern, filename) and filename.endswith(extension)
            ]
            existing_counters.sort()
            if existing_counters:
                counter = existing_counters[-1] + 1
            else:
                counter = 1
            if delimiter:
                filename = f"{prefix}{delimiter}{counter:0{number_padding}}{suffix}{extension}"
            else:
                filename = f"{prefix}{counter:0{number_padding}}{suffix}{extension}"

            while os.path.exists(os.path.join(path, filename)):
                counter += 1
                if delimiter:
                    filename = f"{prefix}{delimiter}{counter:0{number_padding}}{suffix}{extension}"
                else:
                    filename = f"{prefix}{counter:0{number_padding}}{suffix}{extension}"

        return filename

    def write_text_file(self, file, content, encoding):
        try:
            with open(file, 'w', encoding=encoding, newline='\n') as f:
                f.write(content)
        except OSError:
            print(f"Unable to save file `{file}`")

class LoadImagesFromDirBatch_VL:
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
                "start_index": ("INT", {"default": 0, "min": -1, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (sort_methods,),
                "filename_text_extension": (["true", "false"], {"default": "false"}),
                "traverse_subdirs": ("BOOLEAN", {"default": False, "tooltip": "If enabled, will traverse all subdirectories and return full absolute paths. Useful when organizing images in folder structures."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "COUNT", "FILE_NAME")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "load_images"

    CATEGORY = "VL-Nodes/Batch"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if "load_always" in kwargs and kwargs["load_always"]:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_method=None, filename_text_extension="false", traverse_subdirs=False):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.'")

        # Get files based on traverse_subdirs option
        if traverse_subdirs:
            # Traverse all subdirectories
            dir_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    dir_files.append(os.path.join(root, file))
        else:
            # Only current directory
            dir_files = [os.path.join(directory, f) for f in os.listdir(directory)]

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sort_by(dir_files, directory, sort_method)

        # For subdirectory traversal, we already have full paths, otherwise join with directory
        if not traverse_subdirs:
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
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
                has_non_empty_mask = True
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)

            # Handle file path output based on traverse_subdirs option
            filename = os.path.abspath(image_path) if traverse_subdirs else os.path.basename(image_path)
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
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(image1.shape[1], image1.shape[2]), mode="bilinear", align_corners=False)
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


class LoadImagesFromDirList_VL:
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
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (sort_methods,),
                "filename_text_extension": (["true", "false"], {"default": "false"}),
                "traverse_subdirs": ("BOOLEAN", {"default": False, "tooltip": "If enabled, will traverse all subdirectories and return full absolute paths. Useful when organizing images in folder structures."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "FILE_NAME")
    OUTPUT_IS_LIST = (True, True, True)

    FUNCTION = "load_images"

    CATEGORY = "VL-Nodes/Batch"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if "load_always" in kwargs and kwargs["load_always"]:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_method=None, filename_text_extension="false", traverse_subdirs=False):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.'")

        # Get files based on traverse_subdirs option
        if traverse_subdirs:
            # Traverse all subdirectories
            dir_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    dir_files.append(os.path.join(root, file))
        else:
            # Only current directory
            dir_files = [os.path.join(directory, f) for f in os.listdir(directory)]

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sort_by(dir_files, directory, sort_method)
 
        # For subdirectory traversal, we already have full paths, otherwise join with directory
        if not traverse_subdirs:
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

            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
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


class VLNodesFreeMemoryAPI:
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
        print("VLNodesFreeMemory: Clearing all models from memory.")

        # Unload GGUF models first, as they have special handling
        unload_all_gguf_models()

        # Unload InternVL models
        unload_all_internvl_models()

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
            print(f"FreeMemoryAPI: Calling ComfyUI API to free memory: {url}")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print("FreeMemoryAPI: Successfully freed models and execution cache.")
            if self.add_waiting:
                time.sleep(1)
        except Exception as e:
            print(f"FreeMemoryAPI: Failed to call /free API: {e}")

        return (anything,)


NODE_CLASS_MAPPINGS = {
    "VLNodesFreeMemoryAPI": VLNodesFreeMemoryAPI,
    "LoadImagesFromDirBatch_VL": LoadImagesFromDirBatch_VL,
    "LoadImagesFromDirList_VL": LoadImagesFromDirList_VL,
    "TextSave_VL": TextSave_VL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLNodesFreeMemoryAPI": "Free Memory (VL Nodes)",
    "LoadImagesFromDirBatch_VL": "Load Images From Dir (Batch) [VL]",
    "LoadImagesFromDirList_VL": "Load Images From Dir (List) [VL]",
    "TextSave_VL": "Save Text Files [VL]",
}
