import base64
import gc
import os
from io import BytesIO

import torch
from huggingface_hub import snapshot_download
from keye_vl_utils import process_vision_info
from transformers import AutoModel, AutoProcessor, set_seed
from transformers.utils.quantization_config import BitsAndBytesConfig

import comfy.model_management
import folder_paths

from ..utils import find_local_unet_models, hash_seed, tensor2pil

try:
    import flash_attn

    flash_attn_available = True
except ImportError:
    flash_attn_available = False

# Create a directory for Keye models
keye_dir = os.path.join(folder_paths.get_folder_paths("unet")[0], "Keye-VL-HF")
os.makedirs(keye_dir, exist_ok=True)

_keye_loader_instances = []


def unload_all_keye_models():
    """Unloads all Keye models and releases their resources."""
    global _keye_loader_instances
    if not _keye_loader_instances:
        return

    print("Unloading all Keye models...")
    # Create a copy of the list to avoid modification during iteration
    for loader in _keye_loader_instances[:]:
        loader.unload()


class KeyeModelLoader:
    def __init__(self):
        global _keye_loader_instances
        self.model = None
        self.processor = None
        self.cached_params = {}
        if self not in _keye_loader_instances:
            _keye_loader_instances.append(self)

    def __del__(self):
        self.unload()
        if self in _keye_loader_instances:
            _keye_loader_instances.remove(self)

    @classmethod
    def INPUT_TYPES(cls):
        # For now, we only support the specific model mentioned.
        # The structure allows for easy expansion.
        hf_models = ["Kwai-Keye/Keye-VL-8B-Preview"]
        local_models, _ = find_local_unet_models("Keye-VL")
        all_model_options = sorted(list(set(local_models + hf_models)))

        default_model = "Kwai-Keye/Keye-VL-8B-Preview"

        inputs = {
            "required": {
                "model_name": (all_model_options, {"default": default_model}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "precision": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": (["enable", "disable"], {"default": "enable"}),
                "min_pixels": ("INT", {"default": 256 * 28 * 28, "min": 0, "tooltip": "Min pixels for processor"}),
                "max_pixels": ("INT", {"default": 1280 * 28 * 28, "min": 0, "tooltip": "Max pixels for processor"}),
            }
        }

        if flash_attn_available:
            inputs["required"]["use_flash_attention_2"] = (["enable", "disable"], {"default": "enable"})

        return inputs

    RETURN_TYPES = ("KEYE_MODEL",)
    RETURN_NAMES = ("keye_model",)
    FUNCTION = "load_model"
    CATEGORY = "VL-Nodes/Keye-VL"

    def unload(self):
        """Unloads the Keye model and releases associated resources."""
        if self.model is None:
            return

        print("Keye-VL: Unloading model.")

        is_quantized = getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False)

        if is_quantized:
            print("Keye-VL: De-initializing quantized model to free VRAM...")
            try:
                import bitsandbytes as bnb

                def _replace_with_empty(module):
                    for name, child in module.named_children():
                        if isinstance(child, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                            # Deleting the weight and bias from the child module
                            # can help trigger deallocation.
                            if hasattr(child, "weight"):
                                del child.weight
                            if hasattr(child, "bias") and child.bias is not None:
                                del child.bias

                            # Replacing with an empty module to break references
                            setattr(module, name, torch.nn.Module())
                        else:
                            _replace_with_empty(child)

                _replace_with_empty(self.model)
                print("Keye-VL: Replaced quantized layers with empty modules to aid memory release.")
            except Exception as e:
                print(f"Keye-VL: Could not perform deep clean of quantized model, proceeding with standard unload. Error: {e}")
        else:
            try:
                # For regular models, move to CPU to ensure VRAM is released before deleting
                # Check if model can be moved (not offloaded by accelerate)
                can_move_model = True
                try:
                    # Check if model has any modules offloaded to cpu or disk
                    if hasattr(self.model, "hf_device_map"):
                        for module_name, module_device in self.model.hf_device_map.items():
                            if module_device in ["cpu", "disk"]:
                                can_move_model = False
                                break
                except Exception:
                    # If we can't check, assume we can move the model
                    pass

                if can_move_model:
                    self.model.to(device="cpu")
                else:
                    print("Keye-VL: Model has modules offloaded by accelerate, skipping move to CPU")
            except Exception as e:
                print(f"Keye-VL: Warning - could not move model to CPU: {e}")

        del self.model
        del self.processor
        self.model = None
        self.processor = None

        self.cached_params = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def download_model(self, model_name):
        local_dir = os.path.join(keye_dir, model_name.split("/")[-1])
        print(f"Downloading Keye-VL model: {model_name} to {local_dir}")
        try:
            snapshot_download(repo_id=model_name, local_dir=local_dir)
            print(f"Successfully downloaded {model_name} to {local_dir}")
            return local_dir
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise RuntimeError(f"Failed to download model {model_name}. Error: {str(e)}") from e

    def save_quantized_model(self, save_directory):
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        print(f"Quantized model saved to {save_directory}")

    def load_model(self, model_name, quantization, precision, device, auto_download, min_pixels, max_pixels, **kwargs):
        use_flash_attention_2 = kwargs.get("use_flash_attention_2", "enable")
        current_params = {
            "model_name": model_name,
            "quantization": quantization,
            "precision": precision,
            "device": device,
            "auto_download": auto_download,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
        if flash_attn_available:
            current_params["use_flash_attention_2"] = use_flash_attention_2

        if self.model is not None and self.cached_params == current_params:
            print("Keye-VL: Reusing cached model.")
            return ({"model": self.model, "processor": self.processor, "device": device, "dtype": self.model.dtype},)

        if self.model is not None:
            print("Keye-VL: Parameters changed, unloading old model.")
            self.unload()

        print(f"Loading Keye-VL model: {model_name}")

        if precision == "bfloat16":
            dtype = torch.bfloat16
        elif precision == "float16":
            dtype = torch.float16
        elif precision == "float32":
            dtype = torch.float32
        else:  # auto
            dtype = "auto"

        is_repo_id = "/" in model_name

        # Handle quantization
        has_quant_config = False
        # Check for pre-quantization only if it's a local directory
        if not is_repo_id:
            search_name = model_name.split("/")[-1]
            local_models, local_model_paths = find_local_unet_models("Keye-VL")
            path_map = {name: path for name, path in zip(local_models, local_model_paths)}

            model_path_to_check = None
            if search_name in path_map:
                model_path_to_check = path_map[search_name]
            elif os.path.isdir(os.path.join(keye_dir, search_name)):
                model_path_to_check = os.path.join(keye_dir, search_name)

            if model_path_to_check and os.path.isdir(model_path_to_check):
                config_path = os.path.join(model_path_to_check, "config.json")
                if os.path.exists(config_path):
                    import json

                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    if "quantization_config" in config_data:
                        has_quant_config = True
                        print("Keye-VL: Detected pre-quantized model. Ignoring UI quantization setting.")

        model_path_to_load = None

        search_name = model_name.split("/")[-1]
        local_models, local_model_paths = find_local_unet_models("Keye-VL")
        path_map = {name: path for name, path in zip(local_models, local_model_paths)}

        if search_name in path_map:
            model_path_to_load = path_map[search_name]
            print(f"Keye-VL: Found local model at: {model_path_to_load}")
        elif os.path.isdir(os.path.join(keye_dir, search_name)):
            model_path_to_load = os.path.join(keye_dir, search_name)
            print(f"Keye-VL: Found model in custom directory: {model_path_to_load}")

        if model_path_to_load is None and is_repo_id:
            if auto_download == "enable":
                print(f"Keye-VL: Local model not found. Downloading from Hugging Face repo: {model_name}")
                model_path_to_load = self.download_model(model_name)
            else:
                model_path_to_load = model_name
                print(f"Keye-VL: Will attempt to load from Hugging Face repo: {model_path_to_load}")

        if model_path_to_load is None:
            raise FileNotFoundError(f"Keye-VL model '{model_name}' not found. Please check the name or enable auto-download.")

        try:
            model_load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }

            if not has_quant_config and quantization != "none":
                print(f"Keye-VL: Applying {quantization} quantization on-the-fly.")
                if quantization == "4bit":
                    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype if dtype != "auto" else torch.bfloat16)
                else:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                model_load_kwargs["quantization_config"] = quant_config

            processor_load_kwargs = {"trust_remote_code": True}

            if min_pixels > 0 and max_pixels > 0:
                processor_load_kwargs["min_pixels"] = min_pixels
                processor_load_kwargs["max_pixels"] = max_pixels

            # Only enable Flash Attention 2 when using CUDA device
            if flash_attn_available and use_flash_attention_2 == "enable" and device != "cpu":
                model_load_kwargs["attn_implementation"] = "flash_attention_2"
                # Ensure we have a proper dtype for Flash Attention 2
                if dtype == "auto":
                    model_load_kwargs["torch_dtype"] = torch.bfloat16

            device_mapping = "cpu" if device == "cpu" else "auto"
            model_load_kwargs["device_map"] = device_mapping

            self.model = AutoModel.from_pretrained(model_path_to_load, **model_load_kwargs).eval()
            self.processor = AutoProcessor.from_pretrained(model_path_to_load, **processor_load_kwargs)

            self.cached_params = current_params

            return ({"model": self.model, "processor": self.processor, "device": device, "dtype": self.model.dtype},)
        except Exception as e:
            print(f"Error loading Keye-VL model: {str(e)}")
            import traceback

            traceback.print_exc()
            raise e


class KeyeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keye_model": ("KEYE_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "special_captioning_token": ("STRING", {"default": "", "multiline": False}),
                "thinking_mode": (["Auto", "Non-Thinking", "Thinking"], {"default": "Auto"}),
                "resolution_control": (["Default", "Min/Max Pixels", "Exact Dimensions"], {"default": "Default"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.01}),
                "do_sample": (["true", "false"], {"default": "false"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "min_pixels": ("INT", {"default": 256 * 28 * 28, "min": 0}),
                "max_pixels": ("INT", {"default": 1280 * 28 * 28, "min": 0}),
                "resized_height": ("INT", {"default": 0, "min": 0}),
                "resized_width": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "thinking", "text_list", "thinking_list")
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "generate_text"
    CATEGORY = "VL-Nodes/Keye-VL"

    def generate_text(self, keye_model, image, prompt, special_captioning_token, thinking_mode, resolution_control, device, max_new_tokens, temperature, top_p, do_sample, seed, **kwargs):
        model_data = keye_model
        model = model_data["model"]
        processor = model_data["processor"]
        dtype = model_data.get("dtype", torch.bfloat16)

        torch.manual_seed(seed)

        # Determine the device for inference based on user selection
        if device == "cuda":
            inference_device = comfy.model_management.get_torch_device()
        else:
            inference_device = torch.device("cpu")

        original_device = model.device

        is_quantized = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

        # Handle device transitions for quantized models
        if is_quantized and original_device.type == "cpu" and device == "cuda":
            print("Warning: Quantized Keye-VL models (4/8-bit) cannot be moved from CPU to CUDA after loading. Model will remain on CPU for inference.")
            inference_device = torch.device("cpu")

        # For non-quantized models, move to the inference device if needed
        model_on_cpu = original_device.type == "cpu"
        target_device_is_cuda = inference_device.type == "cuda"
        gpu_available = target_device_is_cuda and torch.cuda.is_available()

        # Check if model can be moved (not offloaded by accelerate)
        can_move_model = True
        try:
            # Check if model has any modules offloaded to cpu or disk
            if hasattr(model, "hf_device_map"):
                for _, module_device in model.hf_device_map.items():
                    if module_device in ["cpu", "disk"]:
                        can_move_model = False
                        break
        except Exception:
            # If we can't check, assume we can move the model
            pass

        moved_to_target = False
        if model_on_cpu and gpu_available and not is_quantized and can_move_model:
            print(f"Keye-VL: Moving model from {original_device} to {inference_device} for inference.")
            model.to(inference_device)
            moved_to_target = True
        elif not model_on_cpu and device == "cpu" and can_move_model:
            print(f"Keye-VL: Moving model from {original_device} to CPU for inference.")
            model.to(torch.device("cpu"))
            moved_to_target = True
        elif not can_move_model:
            print(f"Keye-VL: Model has modules offloaded by accelerate, using current device {original_device} for inference.")
            inference_device = original_device

        current_inference_device = model.device
        set_seed(hash_seed(seed))
        output_texts = []
        thinking_analysis_texts = []

        try:
            pil_images = tensor2pil(image)
            batch_size = len(pil_images)
            print(f"Keye-VL: Batch size: {batch_size}")

            for i in range(batch_size):
                print(f"Keye-VL: Processing image {i + 1}/{batch_size}")
                pil_image = pil_images[i]

                # Convert PIL image to base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                base64_image = f"data:image/png;base64,{img_str}"

                final_prompt = prompt
                if thinking_mode == "Non-Thinking":
                    final_prompt += "/no_think"
                elif thinking_mode == "Thinking":
                    final_prompt += "/think"

                image_content = {"type": "image", "image": base64_image}

                if resolution_control == "Exact Dimensions":
                    resized_height = kwargs.get("resized_height", 0)
                    resized_width = kwargs.get("resized_width", 0)
                    if resized_height > 0 and resized_width > 0:
                        image_content["resized_height"] = resized_height
                        image_content["resized_width"] = resized_width
                elif resolution_control == "Min/Max Pixels":
                    min_pixels = kwargs.get("min_pixels", 0)
                    max_pixels = kwargs.get("max_pixels", 0)
                    if min_pixels > 0 and max_pixels > 0:
                        image_content["min_pixels"] = min_pixels
                        image_content["max_pixels"] = max_pixels

                messages = [
                    {
                        "role": "user",
                        "content": [
                            image_content,
                            {"type": "text", "text": final_prompt},
                        ],
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                result = process_vision_info(messages)
                # Handle both 2-tuple and 3-tuple returns from process_vision_info
                if isinstance(result, tuple) and len(result) >= 2:
                    image_inputs, video_inputs = result[0], result[1]
                else:
                    raise ValueError(f"Unexpected return value from process_vision_info: {result}")

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(current_inference_device, dtype=dtype)

                do_sample_bool = do_sample.lower() == "true"
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample_bool,
                    "top_p": top_p if do_sample_bool else None,
                    "temperature": temperature if do_sample_bool else None,
                    "use_cache": True,
                }

                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, **gen_kwargs)
                    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                # Process output based on thinking mode
                import re

                thinking_analysis_text = ""

                # Handle thinking mode output
                if thinking_mode == "Thinking":
                    # Extract content from <think> and <answer> tags
                    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
                    answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)

                    if think_match:
                        thinking_analysis_text = think_match.group(1).strip()
                    if answer_match:
                        output_text = answer_match.group(1).strip()
                    else:
                        # If no <answer> tag, remove <think> tag content and use remaining text
                        output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
                elif thinking_mode == "Auto":
                    # Handle auto mode - could have analysis text and/or thinking tags
                    analysis_match = re.search(r"<analysis>.*?</analysis>", output_text, re.DOTALL)
                    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
                    answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)

                    # Extract analysis text if present
                    if analysis_match:
                        thinking_analysis_text = analysis_match.group(0).strip()

                    # Handle thinking tags if present
                    if think_match:
                        # Append thinking content to analysis text
                        if thinking_analysis_text:
                            thinking_analysis_text += "\n\n" + think_match.group(1).strip()
                        else:
                            thinking_analysis_text = think_match.group(1).strip()

                    # Extract answer content if present, otherwise clean up the text
                    if answer_match:
                        output_text = answer_match.group(1).strip()
                    else:
                        # Remove analysis and thinking tags from output_text
                        output_text = re.sub(r"<analysis>.*?</analysis>", "", output_text, flags=re.DOTALL).strip()
                        output_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()

                if special_captioning_token and special_captioning_token.strip():
                    output_text = f"{special_captioning_token.strip()}, {output_text}"

                output_text = " ".join(output_text.split())
                output_texts.append(output_text.strip())
                thinking_analysis_texts.append(thinking_analysis_text.strip())

        except Exception as e:
            print(f"Error generating text with Keye-VL: {str(e)}")
            import traceback

            traceback.print_exc()
            output_texts.append(f"Error: {str(e)}")
            thinking_analysis_texts.append("")
        finally:
            if moved_to_target and not is_quantized:
                print(f"Keye-VL: Moving model back to {original_device}.")
                model.to(original_device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return ("\n\n=============================\n\n".join(output_texts), "\n\n=============================\n\n".join(thinking_analysis_texts), tuple(output_texts), tuple(thinking_analysis_texts))


NODE_CLASS_MAPPINGS = {
    "KeyeModelLoader": KeyeModelLoader,
    "KeyeNode": KeyeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeyeModelLoader": "Load Keye-VL Model",
    "KeyeNode": "Keye-VL Image to Text",
}
