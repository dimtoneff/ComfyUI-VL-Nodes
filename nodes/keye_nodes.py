import os
import gc
import torch
from transformers import AutoModel, AutoProcessor
from huggingface_hub import snapshot_download
import comfy.model_management
import folder_paths
from ..utils import tensor2pil, find_local_unet_models
import base64
from io import BytesIO
from keye_vl_utils import process_vision_info

try:
    import flash_attn
    flash_attn_available = True
except ImportError:
    flash_attn_available = False

# Create a directory for Keye models
keye_dir = os.path.join(
    folder_paths.get_folder_paths("unet")[0], "Keye-VL-HF")
os.makedirs(keye_dir, exist_ok=True)

_keye_loader_instances = []


def unload_all_keye_models():
    """Unloads all Keye models and releases their resources."""
    global _keye_loader_instances
    if not _keye_loader_instances:
        return

    print("Unloading all Keye models...")
    for loader in _keye_loader_instances:
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
                "precision": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": (["enable", "disable"], {"default": "enable"}),
                "min_pixels": ("INT", {"default": 256*28*28, "min": 0, "tooltip": "Min pixels for processor"}),
                "max_pixels": ("INT", {"default": 1280*28*28, "min": 0, "tooltip": "Max pixels for processor"}),
            }
        }

        if flash_attn_available:
            inputs["required"]["use_flash_attention_2"] = (
                ["enable", "disable"], {"default": "enable"})

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

        try:
            self.model.to(device='cpu')
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
        local_dir = os.path.join(keye_dir, model_name.split('/')[-1])
        print(f"Downloading Keye-VL model: {model_name} to {local_dir}")
        try:
            snapshot_download(repo_id=model_name,
                              local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"Successfully downloaded {model_name} to {local_dir}")
            return local_dir
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise RuntimeError(
                f"Failed to download model {model_name}. Error: {str(e)}") from e

    def load_model(self, model_name, precision, device, auto_download, min_pixels, max_pixels, **kwargs):
        use_flash_attention_2 = kwargs.get("use_flash_attention_2", "enable")
        current_params = {
            "model_name": model_name,
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
            return ({"model": self.model, "processor": self.processor, "device": device},)

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

        is_repo_id = '/' in model_name
        model_path_to_load = None

        search_name = model_name.split('/')[-1]
        local_models, local_model_paths = find_local_unet_models("Keye-VL")
        path_map = {name: path for name, path in zip(
            local_models, local_model_paths)}

        if search_name in path_map:
            model_path_to_load = path_map[search_name]
            print(f"Keye-VL: Found local model at: {model_path_to_load}")
        elif os.path.isdir(os.path.join(keye_dir, search_name)):
            model_path_to_load = os.path.join(keye_dir, search_name)
            print(
                f"Keye-VL: Found model in custom directory: {model_path_to_load}")

        if model_path_to_load is None and is_repo_id:
            if auto_download == "enable":
                print(
                    f"Keye-VL: Local model not found. Downloading from Hugging Face repo: {model_name}")
                model_path_to_load = self.download_model(model_name)
            else:
                model_path_to_load = model_name
                print(
                    f"Keye-VL: Will attempt to load from Hugging Face repo: {model_path_to_load}")

        if model_path_to_load is None:
            raise FileNotFoundError(
                f"Keye-VL model '{model_name}' not found. Please check the name or enable auto-download.")

        try:
            model_load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }

            processor_load_kwargs = {
                "trust_remote_code": True
            }

            if min_pixels > 0 and max_pixels > 0:
                processor_load_kwargs["min_pixels"] = min_pixels
                processor_load_kwargs["max_pixels"] = max_pixels

            if flash_attn_available and use_flash_attention_2 == "enable":
                model_load_kwargs["attn_implementation"] = "flash_attention_2"

            device_mapping = "cpu" if device == "cpu" else "auto"
            model_load_kwargs["device_map"] = device_mapping

            self.model = AutoModel.from_pretrained(
                model_path_to_load, **model_load_kwargs).eval()
            self.processor = AutoProcessor.from_pretrained(
                model_path_to_load, **processor_load_kwargs)

            self.cached_params = current_params

            return ({"model": self.model, "processor": self.processor, "device": device},)
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
                "thinking_mode": (["Auto", "Non-Thinking", "Thinking"], {"default": "Auto"}),
                "resolution_control": (["Default", "Min/Max Pixels", "Exact Dimensions"], {"default": "Default"}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.01}),
                "do_sample": (["true", "false"], {"default": "false"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "min_pixels": ("INT", {"default": 256*28*28, "min": 0}),
                "max_pixels": ("INT", {"default": 1280*28*28, "min": 0}),
                "resized_height": ("INT", {"default": 0, "min": 0}),
                "resized_width": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_text"
    CATEGORY = "VL-Nodes/Keye-VL"

    def generate_text(self, keye_model, image, prompt, thinking_mode, resolution_control, max_new_tokens, temperature, top_p, do_sample, seed, **kwargs):
        model_data = keye_model
        model = model_data["model"]
        processor = model_data["processor"]

        torch.manual_seed(seed)

        inference_device = comfy.model_management.get_torch_device()
        original_device = model.device

        model_on_cpu = original_device.type == 'cpu'
        gpu_available = inference_device.type == 'cuda'

        moved_to_gpu = False
        if model_on_cpu and gpu_available:
            print(
                f"Keye-VL: Moving model from {original_device} to {inference_device} for inference.")
            model.to(inference_device)
            moved_to_gpu = True

        current_inference_device = model.device
        output_text = ""

        try:
            pil_image = tensor2pil(image)

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

            messages = [{
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": final_prompt},
                ],
            }]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs, _ = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(current_inference_device)

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
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

        except Exception as e:
            print(f"Error generating text with Keye-VL: {str(e)}")
            import traceback
            traceback.print_exc()
            output_text = f"Error: {str(e)}"
        finally:
            if moved_to_gpu:
                print(f"Keye-VL: Moving model back to {original_device}.")
                model.to(original_device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return (output_text.strip(),)


NODE_CLASS_MAPPINGS = {
    "KeyeModelLoader": KeyeModelLoader,
    "KeyeNode": KeyeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeyeModelLoader": "Load Keye-VL Model",
    "KeyeNode": "Keye-VL Image to Text",
}
