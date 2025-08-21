import os
import re
import torch
import gc
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from torch.amp.autocast_mode import autocast
import comfy.model_management
import folder_paths
from ..utils import tensor2pil, resize_pil_image, find_local_unet_models

# Create a directory for Ovis-2.5 models
ovis25_dir = os.path.join(
    folder_paths.get_folder_paths("unet")[0], "Ovis2.5-HF")
os.makedirs(ovis25_dir, exist_ok=True)

_ovis25_loader_instances = []


def unload_all_ovis25_models():
    """Unloads all Ovis-2.5 models and releases their resources."""
    global _ovis25_loader_instances
    if not _ovis25_loader_instances:
        return

    print("Unloading all Ovis-2.5 models...")
    for loader in _ovis25_loader_instances:
        loader.unload()


class Ovis25ModelLoader:
    def __init__(self):
        global _ovis25_loader_instances
        self.model = None
        self.text_tokenizer = None
        self.cached_params = {}
        if self not in _ovis25_loader_instances:
            _ovis25_loader_instances.append(self)

    def __del__(self):
        self.unload()
        if self in _ovis25_loader_instances:
            _ovis25_loader_instances.remove(self)

    @classmethod
    def INPUT_TYPES(cls):
        local_models, _ = find_local_unet_models("ovis2.5")
        hf_models = ["AIDC-AI/Ovis2.5-2B", "AIDC-AI/Ovis2.5-9B"]
        all_model_options = sorted(list(set(local_models + hf_models)))

        default_model = "AIDC-AI/Ovis2.5-2B"

        inputs = {
            "required": {
                "model_name": (all_model_options, {"default": default_model}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": (["enable", "disable"], {"default": "enable"}),
            }
        }

        return inputs

    RETURN_TYPES = ("OVIS25_MODEL",)
    RETURN_NAMES = ("ovis25_model",)
    FUNCTION = "load_model"
    CATEGORY = "VL-Nodes/Ovis-2.5"

    def unload(self):
        """Unloads the Ovis-2.5 model and releases associated resources."""
        if self.model is None:
            return

        print("Ovis-2.5: Unloading model.")

        try:
            self.model.to(device='cpu')
        except Exception as e:
            print(f"Ovis-2.5: Warning - could not move model to CPU: {e}")

        del self.model
        del self.text_tokenizer
        self.model = None
        self.text_tokenizer = None

        self.cached_params = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def download_model(self, model_name):
        local_dir = os.path.join(ovis25_dir, model_name.split('/')[-1])
        print(f"Downloading Ovis-2.5 model: {model_name} to {local_dir}")
        try:
            snapshot_download(repo_id=model_name, local_dir=local_dir)
            print(f"Successfully downloaded {model_name} to {local_dir}")
            return local_dir
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise RuntimeError(
                f"Failed to download model {model_name}. Error: {str(e)}")

    def load_model(self, model_name, precision, device, auto_download):
        current_params = {
            "model_name": model_name,
            "precision": precision,
            "device": device,
            "auto_download": auto_download,
        }

        if self.model is not None and self.cached_params == current_params:
            print("Ovis-2.5: Reusing cached model.")
            return ({"model": self.model, "text_tokenizer": self.text_tokenizer, "device": device, "dtype": self.model.dtype},)

        if self.model is not None:
            print("Ovis-2.5: Parameters changed, unloading old model.")
            self.unload()

        print(f"Loading Ovis-2.5 model: {model_name}")

        if precision == "bfloat16":
            dtype = torch.bfloat16
        elif precision == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        is_repo_id = '/' in model_name
        model_path_to_load = None

        # Determine the name to search for locally.
        search_name = model_name.split('/')[-1]

        # Use the utility to find all local models and their paths.
        local_models, local_model_paths = find_local_unet_models("ovis2.5")
        path_map = {name: path for name, path in zip(
            local_models, local_model_paths)}

        if search_name in path_map:
            model_path_to_load = path_map[search_name]
            print(f"Ovis-2.5: Found local model at: {model_path_to_load}")

        # If not found locally and it's a repo_id, handle download or direct HF loading.
        if model_path_to_load is None and is_repo_id:
            if auto_download == "enable":
                print(
                    f"Ovis-2.5: Local model not found. Downloading from Hugging Face repo: {model_name}")
                model_path_to_load = self.download_model(model_name)
            else:
                model_path_to_load = model_name
                print(
                    f"Ovis-2.5: Will attempt to load from Hugging Face repo: {model_path_to_load}")

        if model_path_to_load is None:
            raise FileNotFoundError(
                f"Ovis-2.5 model '{model_name}' not found in any of the 'unet' directories: {folder_paths.get_folder_paths('unet')}")

        try:
            load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }

            device_mapping = "cpu" if device == "cpu" else "auto"
            load_kwargs["device_map"] = device_mapping

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_to_load, **load_kwargs).eval()
            self.text_tokenizer = self.model.text_tokenizer
            self.cached_params = current_params

            return ({"model": self.model, "text_tokenizer": self.text_tokenizer, "device": device, "dtype": dtype},)
        except Exception as e:
            print(f"Error loading Ovis-2.5 model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


class Ovis25ImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ovis25_model": ("OVIS25_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "resize_image": ("BOOLEAN", {"default": True, "label_on": "Resize Enabled", "label_off": "Resize Disabled"}),
                "enable_thinking": ("BOOLEAN", {"default": True}),
                "enable_thinking_budget": ("BOOLEAN", {"default": True}),
                "max_new_tokens": ("INT", {"default": 3072, "min": 64, "max": 8192, "tooltip": "Max new tokens for the response. Must be > thinking_budget + 25."}),
                "thinking_budget": ("INT", {"default": 2048, "min": 64, "max": 8192, "tooltip": "Token budget for the thinking process."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.01}),
                "do_sample": (["true", "false"], {"default": "false"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "thinking")
    FUNCTION = "generate_text"
    CATEGORY = "VL-Nodes/Ovis-2.5"

    def generate_text(self, ovis25_model, image, prompt, resize_image, enable_thinking, enable_thinking_budget, max_new_tokens, thinking_budget, temperature, top_p, do_sample):
        model_data = ovis25_model
        model = model_data["model"]
        text_tokenizer = model_data["text_tokenizer"]
        dtype = model_data["dtype"]

        inference_device = comfy.model_management.get_torch_device()
        original_device = model.device

        model_on_cpu = original_device.type == 'cpu'
        gpu_available = inference_device.type == 'cuda'

        moved_to_gpu = False
        if model_on_cpu and gpu_available:
            print(
                f"Ovis-2.5: Moving model from {original_device} to {inference_device} for inference.")
            model.to(inference_device)
            moved_to_gpu = True

        current_inference_device = model.device
        thinking_text = ""
        final_text = ""

        try:
            pil_image = tensor2pil(image)
            if resize_image:
                pil_image = resize_pil_image(
                    pil_image, target_size=512, node_name="Ovis25ImageToText")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }]

            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )

            input_ids = input_ids.to(device=current_inference_device)
            pixel_values = pixel_values.to(
                device=current_inference_device, dtype=dtype) if pixel_values is not None else None
            grid_thws = grid_thws.to(
                device=current_inference_device) if grid_thws is not None else None

            do_sample_bool = do_sample.lower() == "true"

            gen_kwargs = {
                "enable_thinking": enable_thinking,
                "enable_thinking_budget": enable_thinking_budget if enable_thinking else False,
                "max_new_tokens": max_new_tokens,
                "thinking_budget": thinking_budget,
                "do_sample": do_sample_bool,
                "top_p": top_p if do_sample_bool else None,
                "temperature": temperature if do_sample_bool else None,
                "eos_token_id": text_tokenizer.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True,
            }

            with torch.inference_mode(), autocast(device_type=current_inference_device.type, dtype=dtype):
                outputs = model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    **gen_kwargs
                )
                response = text_tokenizer.decode(
                    outputs[0], skip_special_tokens=True)

            think_match = re.search(
                r'<think>(.*?)</think>', response, re.DOTALL)
            if think_match:
                thinking_text = think_match.group(1).strip()
                final_text = response.replace(think_match.group(0), '').strip()
            else:
                final_text = response.strip()

        except Exception as e:
            print(f"Error generating text with Ovis-2.5: {str(e)}")
            import traceback
            traceback.print_exc()
            thinking_text = f"Error: {str(e)}"
        finally:
            if moved_to_gpu:
                print(f"Ovis-2.5: Moving model back to {original_device}.")
                model.to(original_device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return (final_text.strip('"'), thinking_text,)


NODE_CLASS_MAPPINGS = {
    "Ovis25ModelLoader": Ovis25ModelLoader,
    "Ovis25ImageToText": Ovis25ImageToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ovis25ModelLoader": "Load Ovis-2.5 Model",
    "Ovis25ImageToText": "Ovis-2.5 Image to Text",
}
