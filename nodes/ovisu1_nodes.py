
import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.utils.quantization_config import BitsAndBytesConfig
from huggingface_hub import snapshot_download
from torch.amp.autocast_mode import autocast
import comfy.model_management
import folder_paths
from ..utils import tensor2pil, resize_pil_image, find_local_unet_models

# Create a directory for Ovis-U1 models
ovis_u1_dir = os.path.join(
    folder_paths.get_folder_paths("unet")[0], "Ovis-U1-HF")
os.makedirs(ovis_u1_dir, exist_ok=True)

_ovisu1_loader_instances = []


def unload_all_ovisu1_models():
    """Unloads all Ovis-U1 models and releases their resources."""
    global _ovisu1_loader_instances
    if not _ovisu1_loader_instances:
        return

    print("Unloading all Ovis-U1 models...")
    for loader in _ovisu1_loader_instances:
        loader.unload()


class OvisU1ModelLoader:
    def __init__(self):
        global _ovisu1_loader_instances
        self.model = None
        self.text_tokenizer = None
        self.visual_tokenizer = None
        self.cached_params = {}
        if self not in _ovisu1_loader_instances:
            _ovisu1_loader_instances.append(self)

    def __del__(self):
        self.unload()
        if self in _ovisu1_loader_instances:
            _ovisu1_loader_instances.remove(self)

    @classmethod
    def INPUT_TYPES(cls):
        local_models, _ = find_local_unet_models("ovis-u1")
        # Always include the option to download the base model from Hugging Face.
        hf_model_id = "AIDC-AI/Ovis-U1-3B"
        all_model_options = sorted(list(set(local_models + [hf_model_id])))

        default_model = hf_model_id
        if "Ovis-U1-3B-4bit" in all_model_options:
            default_model = "Ovis-U1-3B-4bit"

        return {
            "required": {
                "model_name": (all_model_options, {"default": default_model}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("OVIS_U1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "VL-Nodes/Ovis-U1"

    def unload(self):
        """Unloads the Ovis-U1 model and releases associated resources."""
        if self.model is None:
            return

        print("Ovis-U1: Unloading model.")

        is_quantized = getattr(self.model, 'is_loaded_in_8bit', False) or getattr(
            self.model, 'is_loaded_in_4bit', False)

        if is_quantized:
            print("Ovis-U1: De-initializing quantized model to free VRAM...")
            try:
                import bitsandbytes as bnb

                def _replace_with_empty(module):
                    for name, child in module.named_children():
                        if isinstance(child, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                            # Deleting the weight and bias from the child module
                            # can help trigger deallocation.
                            if hasattr(child, 'weight'):
                                del child.weight
                            if hasattr(child, 'bias') and child.bias is not None:
                                del child.bias

                            # Replacing with an empty module to break references
                            setattr(module, name, torch.nn.Module())
                        else:
                            _replace_with_empty(child)

                _replace_with_empty(self.model)
                print(
                    "Ovis-U1: Replaced quantized layers with empty modules to aid memory release.")
            except Exception as e:
                print(
                    f"Ovis-U1: Could not perform deep clean of quantized model, proceeding with standard unload. Error: {e}")
        else:
            try:
                # For regular models, move to CPU to ensure VRAM is released before deleting
                self.model.to(device='cpu')
            except Exception as e:
                print(f"Ovis-U1: Warning - could not move model to CPU: {e}")

        del self.model
        del self.text_tokenizer
        del self.visual_tokenizer
        self.model = None
        self.text_tokenizer = None
        self.visual_tokenizer = None

        self.cached_params = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def download_model(self, model_name):
        """Download the model files from Hugging Face if they don't exist locally."""
        local_dir = os.path.join(ovis_u1_dir, model_name.split('/')[-1])

        print(f"Downloading Ovis-U1 model: {model_name} to {local_dir}")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=local_dir
            )
            print(f"Successfully downloaded {model_name} to {local_dir}")
            return local_dir
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            raise RuntimeError(
                f"Failed to download model {model_name}. Error: {str(e)}")

    def load_model(self, model_name, quantization, precision, device, auto_download):
        current_params = {
            "model_name": model_name,
            "quantization": quantization,
            "precision": precision,
            "device": device,
        }
        if self.model is not None and self.cached_params == current_params:
            print("Ovis-U1: Reusing cached model.")
            return ({"model": self.model, "text_tokenizer": self.text_tokenizer, "visual_tokenizer": self.visual_tokenizer, "device": device, "dtype": self.model.dtype},)

        if self.model is not None:
            print("Ovis-U1: Parameters changed, unloading old model.")
            self.unload()

        # Monkey-patch AutoTokenizer.from_pretrained to force trust_remote_code=True
        # This is a workaround for models that call this function internally without propagating the argument,
        # which causes an interactive prompt that fails on Windows due to a missing signal (SIGALRM).
        original_from_pretrained = AutoTokenizer.from_pretrained

        def patched_from_pretrained_impl(cls, pretrained_model_name_or_path, *inputs, **kwargs):
            print(
                "OvisU1-Loader: Using patched AutoTokenizer.from_pretrained to force trust_remote_code=True")
            kwargs['trust_remote_code'] = True
            # The original method is already bound to the class, so we call it directly.
            return original_from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        # The original from_pretrained is a classmethod, so we need to wrap our patch similarly.
        AutoTokenizer.from_pretrained = classmethod(
            patched_from_pretrained_impl)

        # Monkey-patch CONFIG_MAPPING.register to handle re-registration of 'aimv2'
        # The Ovis-U1 model's remote code tries to register 'aimv2' every time it's loaded,
        # causing an error on subsequent runs. This patch makes the registration idempotent.
        original_config_register = CONFIG_MAPPING.register

        def patched_config_register(key, value, exist_ok=False):
            if key == 'aimv2':
                print(
                    "OvisU1-Loader: Patching CONFIG_MAPPING.register for 'aimv2' with exist_ok=True")
                return original_config_register(key, value, exist_ok=True)
            return original_config_register(key, value, exist_ok=exist_ok)
        CONFIG_MAPPING.register = patched_config_register

        print(f"Loading Ovis-U1 model: {model_name}")

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
        local_models, local_model_paths = find_local_unet_models("ovis-u1")
        path_map = {name: path for name, path in zip(
            local_models, local_model_paths)}

        if search_name in path_map:
            model_path_to_load = path_map[search_name]
            print(f"OvisU1: Found local model at: {model_path_to_load}")

        # If not found locally and it's a repo_id, handle download or direct HF loading.
        if model_path_to_load is None and is_repo_id:
            if auto_download == "enable":
                print(
                    f"OvisU1: Local model not found. Downloading from Hugging Face repo: {model_name}")
                model_path_to_load = self.download_model(model_name)
            else:
                model_path_to_load = model_name
                print(
                    f"OvisU1: Will attempt to load from Hugging Face repo: {model_path_to_load}")

        if model_path_to_load is None:
            raise FileNotFoundError(
                f"Ovis-U1 model '{model_name}' not found in any of the 'unet' directories: {folder_paths.get_folder_paths('unet')}")
        try:
            load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
                "output_loading_info": True,
            }

            has_quant_config = False
            # Check for pre-quantization only if it's a local directory
            if os.path.isdir(model_path_to_load):
                config_path = os.path.join(model_path_to_load, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    if 'quantization_config' in config_data:
                        has_quant_config = True
                        print(
                            "OvisU1: Detected pre-quantized model. Ignoring UI quantization setting.")

            if not has_quant_config and quantization != "none":
                print(
                    f"OvisU1: Applying {quantization} quantization on-the-fly.")
                if quantization == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=dtype
                    )
                else:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["quantization_config"] = quant_config

            device_mapping = "cpu" if device == "cpu" else "auto"
            print(f"OvisU1: Using device_map='{device_mapping}'")
            load_kwargs["device_map"] = device_mapping

            model, loading_info = AutoModelForCausalLM.from_pretrained(
                model_path_to_load,
                **load_kwargs
            )

            print(f'Loading info of Ovis-U1:\n{loading_info}')
            self.model = model.eval()
            self.text_tokenizer = model.get_text_tokenizer()
            self.visual_tokenizer = model.get_visual_tokenizer()
            self.cached_params = current_params

            return ({"model": self.model, "text_tokenizer": self.text_tokenizer, "visual_tokenizer": self.visual_tokenizer, "device": device, "dtype": dtype},)
        except Exception as e:
            print(f"Error loading Ovis-U1 model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
        finally:
            # Restore the original method to avoid side effects on other nodes.
            AutoTokenizer.from_pretrained = original_from_pretrained
            print("OvisU1-Loader: Restored original AutoTokenizer.from_pretrained")
            CONFIG_MAPPING.register = original_config_register
            print("OvisU1-Loader: Restored original CONFIG_MAPPING.register")


class OvisU1ImageCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS_U1_MODEL",),
                "image": ("IMAGE",),
                "resize_image": ("BOOLEAN", {"default": True, "label_on": "Resize Enabled", "label_off": "Resize Disabled"}),
                "prompt": ("STRING", {"default": "Describe this image in detail.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 75, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.01}),
                "do_sample": (["true", "false"], {"default": "false"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "generate_caption"
    CATEGORY = "VL-Nodes/Ovis-U1"

    def generate_caption(self, model, image, resize_image, prompt, max_new_tokens, temperature, top_p, do_sample):
        model_data = model
        model = model_data["model"]
        text_tokenizer = model_data["text_tokenizer"]
        dtype = model_data["dtype"]

        # Determine the device for inference (e.g., 'cuda' or 'cpu')
        inference_device = comfy.model_management.get_torch_device()
        original_device = model.device

        is_quantized = getattr(model, 'is_loaded_in_8bit', False) or getattr(
            model, 'is_loaded_in_4bit', False)

        if is_quantized and original_device.type == 'cpu':
            raise RuntimeError(
                "Quantized Ovis-U1 models (4/8-bit) cannot be moved after loading. "
                "Please use the OvisU1ModelLoader with the 'device' set to 'cuda' to load it directly to VRAM."
            )

        # For non-quantized models, move to the inference device if it's on CPU
        # and a GPU is available.
        model_on_cpu = original_device.type == 'cpu'
        gpu_available = inference_device.type == 'cuda'

        moved_to_gpu = False
        if model_on_cpu and gpu_available and not is_quantized:
            print(
                f"Ovis-U1 Caption: Moving model from {original_device} to {inference_device} for inference.")
            model.to(inference_device)
            moved_to_gpu = True

        # The actual device for inference is where the model is now
        current_inference_device = model.device

        try:
            pil_image = tensor2pil(image)

            if resize_image:
                pil_image = resize_pil_image(
                    pil_image, target_size=512, node_name="OvisU1ImageCaption")

            query = f"<image>\n{prompt}"

            # Preprocess inputs
            _, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                query,
                [pil_image],
                generation_preface='',
                return_labels=False,
                propagate_exception=False,
                multimodal_type='single_image',
                fix_sample_overall_length_navit=False
            )
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)

            # Move inputs to the correct device for inference
            input_ids = input_ids.unsqueeze(0).to(
                device=current_inference_device)
            attention_mask = attention_mask.unsqueeze(
                0).to(device=current_inference_device)

            if pixel_values is not None:
                pixel_values = pixel_values.to(
                    device=current_inference_device, dtype=dtype)
            if grid_thws is not None:
                grid_thws = grid_thws.to(device=current_inference_device)

            do_sample_bool = do_sample.lower() == "true"

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample_bool,
                "top_p": top_p if do_sample_bool else None,
                "temperature": temperature if do_sample_bool else None,
                "eos_token_id": text_tokenizer.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True,
            }

            # Use autocast for mixed-precision
            with torch.inference_mode(), autocast(device_type=current_inference_device.type, dtype=dtype):
                output_ids = model.generate(input_ids, pixel_values=pixel_values,
                                            attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)[0]
                gen_text = text_tokenizer.decode(
                    output_ids, skip_special_tokens=True)

            return (gen_text.strip().strip('"'),)
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            import traceback
            traceback.print_exc()
            return (f"Error generating caption: {str(e)}",)
        finally:
            # Move model back to CPU if we moved it
            if moved_to_gpu:
                print(
                    f"Ovis-U1 Caption: Moving model back to {original_device}.")
                model.to(original_device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS = {
    "OvisU1ModelLoader": OvisU1ModelLoader,
    "OvisU1ImageCaption": OvisU1ImageCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OvisU1ModelLoader": "Load Ovis-U1 Model",
    "OvisU1ImageCaption": "Ovis-U1 Image Caption",
}
