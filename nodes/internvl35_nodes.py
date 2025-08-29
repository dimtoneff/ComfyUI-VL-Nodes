import gc
import os
import re

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, set_seed
from transformers.utils.quantization_config import BitsAndBytesConfig

import comfy.model_management
import folder_paths
from ..utils import find_local_unet_models, hash_seed, tensor2pil

try:
    import flash_attn

    flash_attn_available = True
except ImportError:
    flash_attn_available = False

# Create a directory for InternVL models
internvl_dir = os.path.join(folder_paths.get_folder_paths("unet")[0], "InternVL-HF")
os.makedirs(internvl_dir, exist_ok=True)

# --- Preprocessing functions from t.txt ---

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_pil(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# --- End Preprocessing ---

_internvl_loader_instances = []


def unload_all_internvl_models():
    global _internvl_loader_instances
    if not _internvl_loader_instances:
        return
    print("Unloading all InternVL models...")
    for loader in _internvl_loader_instances[:]:
        loader.unload()


class InternVL3_5_ModelLoader:
    def __init__(self):
        global _internvl_loader_instances
        self.model = None
        self.tokenizer = None
        self.cached_params = {}
        if self not in _internvl_loader_instances:
            _internvl_loader_instances.append(self)

    def __del__(self):
        self.unload()
        if self in _internvl_loader_instances:
            _internvl_loader_instances.remove(self)

    @classmethod
    def INPUT_TYPES(cls):
        hf_models = [
            "OpenGVLab/InternVL3_5-1B",
            "OpenGVLab/InternVL3_5-2B",
            "OpenGVLab/InternVL3_5-4B",
            "OpenGVLab/InternVL3_5-8B",
            "OpenGVLab/InternVL3_5-14B",
            "OpenGVLab/InternVL3_5-38B",
        ]
        local_models, _ = find_local_unet_models("InternVL")
        all_model_options = sorted(list(set(local_models + hf_models)))

        inputs = {
            "required": {
                "model_name": (all_model_options, {"default": "OpenGVLab/InternVL3_5-1B"}),
                "quantization": (["none", "8bit", "4bit"], {"default": "none"}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": (["enable", "disable"], {"default": "enable"}),
            }
        }
        if flash_attn_available:
            inputs["required"]["use_flash_attn"] = (["enable", "disable"], {"default": "enable"})
        return inputs

    RETURN_TYPES = ("INTERNVL_MODEL",)
    RETURN_NAMES = ("internvl_model",)
    FUNCTION = "load_model"
    CATEGORY = "VL-Nodes 👁️‍🗨️/InternVL"

    def unload(self):
        if self.model is None:
            return
        print("InternVL: Unloading model.")
        try:
            if hasattr(self.model, "to"):
                self.model.to(device="cpu")
        except Exception as e:
            print(f"InternVL: Warning - could not move model to CPU: {e}")

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self.cached_params = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def download_model(self, model_name):
        local_dir = os.path.join(internvl_dir, model_name.split("/")[-1])
        if os.path.isdir(local_dir):
            print(f"InternVL: Using existing model in {local_dir}")
            return local_dir
        print(f"InternVL: Downloading model: {model_name} to {local_dir}")
        try:
            snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"Successfully downloaded {model_name} to {local_dir}")
            return local_dir
        except Exception as e:
            raise RuntimeError(f"Failed to download model {model_name}. Error: {str(e)}") from e

    def load_model(self, model_name, quantization, precision, device, auto_download, **kwargs):
        use_flash_attn = kwargs.get("use_flash_attn", "enable") == "enable"
        current_params = {
            "model_name": model_name,
            "quantization": quantization,
            "precision": precision,
            "device": device,
            "auto_download": auto_download,
            "use_flash_attn": use_flash_attn,
        }

        if self.model is not None and self.cached_params == current_params:
            print("InternVL: Reusing cached model.")
            return ({"model": self.model, "tokenizer": self.tokenizer, "device": device},)

        if self.model is not None:
            print("InternVL: Parameters changed, unloading old model.")
            self.unload()

        print(f"InternVL: Loading model {model_name}")

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(precision, torch.bfloat16)

        is_repo_id = "/" in model_name
        model_path_to_load = model_name

        if is_repo_id:
            search_name = model_name.split("/")[-1]
            local_models, local_model_paths = find_local_unet_models("InternVL")
            path_map = {name: path for name, path in zip(local_models, local_model_paths)}
            if search_name in path_map:
                model_path_to_load = path_map[search_name]
                print(f"InternVL: Found local model at: {model_path_to_load}")
            elif auto_download == "enable":
                model_path_to_load = self.download_model(model_name)

        model_load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if device == "cpu":
            model_load_kwargs["device_map"] = "cpu"
        else:
            model_load_kwargs["device_map"] = "auto"

        if flash_attn_available and use_flash_attn and device == "cuda":
            model_load_kwargs["use_flash_attn"] = True

        if quantization != "none":
            if quantization == "4bit":
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
            else:  # 8bit
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model_load_kwargs["quantization_config"] = quant_config

        try:
            self.model = AutoModel.from_pretrained(model_path_to_load, **model_load_kwargs).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_to_load, trust_remote_code=True, use_fast=False)
            self.cached_params = current_params
        except Exception as e:
            print(f"InternVL: Error loading model: {e}")
            self.unload()
            raise

        return ({"model": self.model, "tokenizer": self.tokenizer, "device": device},)


class InternVL3_5_ImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "internvl_model": ("INTERNVL_MODEL",),
                "image": ("IMAGE",),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are an AI assistant that rigorously follows this response protocol:\n\n1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and  </think> tags.\n\n2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.\n\nEnsure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.",
                        "multiline": True,
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "You are an assistant who perfectly describes images. Fully describe the image as a prompt for Stable Diffusion text2image SDXL model. If you see nsfw stuff describe it too. Do not omit anything! Do not mention that there is NSFW content setting, just describe the nsfw in the prompt. !Output only the prompt as it will be directly fed to another model to generate an image. Be clear and use short sentences to describe the image! DO not use MD format, do not add titles or paragraphs!",
                        "multiline": True,
                    },
                ),
                "special_captioning_token": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_num_tiles": ("INT", {"default": 12, "min": 1, "max": 24}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "thinking", "text_list", "thinking_list")
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "generate_text"
    CATEGORY = "VL-Nodes 👁️‍🗨️/InternVL"

    def generate_text(self, internvl_model, image, system_prompt, prompt, special_captioning_token, seed, max_num_tiles, max_new_tokens, do_sample, temperature, top_p, top_k):
        model = internvl_model["model"]
        tokenizer = internvl_model["tokenizer"]
        loader_device = internvl_model["device"]

        inference_device = comfy.model_management.get_torch_device() if loader_device == "cuda" else torch.device("cpu")
        original_device = model.device

        moved_to_inference_device = False
        if original_device.type == "cpu" and inference_device.type == "cuda":
            print(f"InternVL: Moving model from {original_device} to {inference_device} for inference.")
            model.to(inference_device)
            moved_to_inference_device = True

        set_seed(hash_seed(seed))
        output_texts = []
        thinking_texts = []

        try:
            pil_images = tensor2pil(image)
            print(f"InternVL: Batch size: {len(pil_images)}")

            for i, pil_image in enumerate(pil_images):
                print(f"InternVL: Processing image {i + 1}/{len(pil_images)}")
                try:
                    pixel_values = load_image_pil(pil_image, max_num=max_num_tiles).to(model.dtype).to(model.device)
                    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "temperature": temperature, "top_p": top_p, "top_k": top_k}

                    # The InternVL model expects an <image> placeholder in the prompt.
                    # If it's not provided by the user, we add it to the beginning.
                    # This allows flexible placement of the image reference within the text.
                    if "<image>" in prompt:
                        final_user_prompt = prompt
                    else:
                        final_user_prompt = "<image>\n" + prompt

                    # Use model.chat directly with the custom system prompt
                    with torch.inference_mode(), autocast(device_type=inference_device.type, dtype=model.dtype):
                        # Set the custom system prompt if provided
                        if system_prompt and system_prompt.strip():
                            model.system_message = system_prompt

                        response = model.chat(tokenizer, pixel_values, final_user_prompt, generation_config)

                    print(f"InternVL: Raw response for image {i + 1}: {response}")
                    thinking_text = ""
                    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                    if think_match:
                        thinking_text = think_match.group(1).strip()
                        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

                    if special_captioning_token and special_captioning_token.strip():
                        response = f"{special_captioning_token.strip()}, {response}"

                    output_texts.append(response.strip())
                    thinking_texts.append(thinking_text)

                except Exception as e:
                    print(f"InternVL: Error during inference for image {i + 1}: {e}")
                    output_texts.append(f"Error: {e}")
                    thinking_texts.append("")

        finally:
            if moved_to_inference_device:
                print(f"InternVL: Moving model back to {original_device}.")
                model.to(original_device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return ("\n\n=============================\n\n".join(output_texts), "\n\n=============================\n\n".join(thinking_texts), tuple(output_texts), tuple(thinking_texts))


NODE_CLASS_MAPPINGS = {
    "InternVL3_5_ModelLoader": InternVL3_5_ModelLoader,
    "InternVL3_5_ImageToText": InternVL3_5_ImageToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InternVL3_5_ModelLoader": "👁️ Load InternVL3.5 Model",
    "InternVL3_5_ImageToText": "👁️ InternVL3.5 Image to Text",
}
