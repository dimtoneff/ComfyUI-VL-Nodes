import os
import gc
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils.quantization_config import BitsAndBytesConfig
import folder_paths
from torch.amp.autocast_mode import autocast
from ..utils import tensor2pil, find_local_unet_models

# Global registry for LFM2 loaders
# _lfm2_gguf_loader_instances = []
_lfm2_hf_loader_instances = []


# def unload_all_lfm2_gguf_models():
#     """Unloads all LFM2 GGUF models and releases their resources."""
#     global _lfm2_gguf_loader_instances
#     if not _lfm2_gguf_loader_instances:
#         return
#     print("Unloading all LFM2 GGUF models...")
#     for loader in _lfm2_gguf_loader_instances:
#         loader.unload()


def unload_all_lfm2_hf_models():
    """Unloads all LFM2 HuggingFace models and releases their resources."""
    global _lfm2_hf_loader_instances
    if not _lfm2_hf_loader_instances:
        return
    print("Unloading all LFM2 HuggingFace models...")
    for loader in _lfm2_hf_loader_instances:
        loader.unload()


# --- Transformers Nodes ---

lfm2_hf_models_dir = os.path.join(
    folder_paths.get_folder_paths("unet")[0], "LFM2-VL-HF")
os.makedirs(lfm2_hf_models_dir, exist_ok=True)


class LFM2TransformerModelLoader:
    def __init__(self):
        global _lfm2_hf_loader_instances
        self.model = None
        self.processor = None
        self.cached_params = {}
        if self not in _lfm2_hf_loader_instances:
            _lfm2_hf_loader_instances.append(self)

    def __del__(self):
        self.unload()
        if self in _lfm2_hf_loader_instances:
            _lfm2_hf_loader_instances.remove(self)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (["LiquidAI/LFM2-VL-450M", "LiquidAI/LFM2-VL-1.6B"],),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("LFM2_HF_MODEL",)
    RETURN_NAMES = ("lfm2_hf_model",)
    FUNCTION = "load_model"
    CATEGORY = "VL-Nodes/LFM2-VL"

    def unload(self):
        if self.model is None:
            return
        print("LFM2 HF: Unloading model.")
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self.cached_params = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def download_model(self, model_id):
        """Download the model from Hugging Face."""
        # The download directory is defined at the module level
        local_dir = os.path.join(lfm2_hf_models_dir, model_id.split('/')[-1])
        print(f"LFM2 HF: Downloading model: {model_id} to {local_dir}")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print(
                f"LFM2 HF: Successfully downloaded {model_id} to {local_dir}")
            return local_dir
        except Exception as e:
            print(f"LFM2 HF: Error downloading model: {str(e)}")
            raise RuntimeError(
                f"Failed to download model {model_id}. Error: {str(e)}")

    def load_model(self, model_id, quantization, precision, device, auto_download):
        current_params = {
            "model_id": model_id,
            "quantization": quantization,
            "precision": precision,
            "device": device,
        }
        if self.model is not None and self.cached_params == current_params:
            print("LFM2 HF: Reusing cached model.")
            return ({"model": self.model, "processor": self.processor, "device": device, "dtype": self.model.dtype},)

        if self.model is not None:
            print("LFM2 HF: Parameters changed, unloading old model.")
            self.unload()

        print(f"Loading LFM2 HF model: {model_id}")

        if precision == "bfloat16":
            dtype = torch.bfloat16
        elif precision == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        model_path_to_load = None
        is_repo_id = '/' in model_id

        # Determine the name to search for locally.
        search_name = model_id.split('/')[-1]

        # Use the utility to find all local models and their paths.
        local_models, local_model_paths = find_local_unet_models("lfm2")
        print(
            f"LFM2 HF: Found local models: {local_models}, {local_model_paths}")
        path_map = {name: path for name, path in zip(
            local_models, local_model_paths)}

        if search_name in path_map:
            model_path_to_load = path_map[search_name]
            print(f"LFM2 HF: Found local model at: {model_path_to_load}")

        # If not found locally and it's a repo_id, handle download or direct HF loading.
        if model_path_to_load is None and is_repo_id:
            if auto_download == "enable":
                print(
                    f"LFM2 HF: Local model not found. Downloading from Hugging Face repo: {model_id}")
                model_path_to_load = self.download_model(model_id)
            else:
                model_path_to_load = model_id
                print(
                    f"LFM2 HF: Will attempt to load from Hugging Face repo: {model_path_to_load}")
        if model_path_to_load is None:
            raise FileNotFoundError(
                f"LFM2-VL model '{model_id}' not found in any of the 'unet' directories: {folder_paths.get_folder_paths('unet')}")
        try:
            load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }

            if quantization != "none":
                if quantization == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=dtype
                    )
                else:  # 8bit
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["quantization_config"] = quant_config

            device_mapping = "cpu" if device == "cpu" else "auto"
            load_kwargs["device_map"] = device_mapping

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path_to_load, **load_kwargs)
            self.processor = AutoProcessor.from_pretrained(
                model_path_to_load, trust_remote_code=True)
            self.cached_params = current_params

            return ({"model": self.model, "processor": self.processor, "device": device, "dtype": dtype},)
        except Exception as e:
            print(f"Error loading LFM2 HF model: {str(e)}")
            raise e


class LFM2TransformerImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lfm2_hf_model": ("LFM2_HF_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe this image.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 75, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1}),
                "min_p": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_text"
    CATEGORY = "VL-Nodes/LFM2-VL"

    def generate_text(self, lfm2_hf_model, image, prompt, max_new_tokens, temperature, min_p, repetition_penalty):
        model = lfm2_hf_model["model"]
        processor = lfm2_hf_model["processor"]
        target_device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = lfm2_hf_model["dtype"]

        original_device = model.device
        target_device = torch.device(target_device_str)

        move_model = target_device.type == 'cuda' and original_device.type == 'cpu'

        if move_model:
            print(
                f"LFM2 HF: Moving model from {original_device} to {target_device} for inference.")
            model.to(target_device)

        try:
            pil_image = tensor2pil(image)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            with torch.inference_mode(), autocast(device_type=target_device.type, dtype=dtype):
                inputs = processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ).to(target_device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True if temperature > 0 else False
                )
                result = processor.batch_decode(
                    outputs, skip_special_tokens=True)[0]
        finally:
            if move_model:
                print(f"LFM2 HF: Moving model back to {original_device}.")
                model.to(original_device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle different assistant markers
        possible_markers = ["<|im_start|>assistant\n", "assistant\n"]
        final_text = result.strip()

        for marker in possible_markers:
            last_occurrence = result.rfind(marker)
            if last_occurrence != -1:
                final_text = result[last_occurrence + len(marker):].strip()
                # Remove <|im_end|> if it exists
                if final_text.endswith("<|im_end|>"):
                    final_text = final_text[:-len("<|im_end|>")].strip()
                break  # Marker found, no need to check others

        if "assistant" in final_text:
            final_text = final_text.split("assistant")[-1].strip()

        return (final_text.strip('"'),)

# --- GGUF Nodes ---
# Loader works with latest llama.cpp but the inference did not work properly.

# class CustomLlava15ChatHandler(Llava15ChatHandler):
#     """
#     Custom Chat Handler that inherits from Llava15ChatHandler and adds
#     proper resource management (__del__ and close) to prevent VRAM leaks.
#     """

#     def close(self):
#         if hasattr(self, '_exit_stack'):
#             self._exit_stack.close()

#     def __del__(self):
#         self.close()


# class LFM2GGUFModelLoader:
#     def __init__(self):
#         global _lfm2_gguf_loader_instances
#         self.llm = None
#         self.chat_handler = None
#         self.initial_state = None
#         self.cached_params = {}
#         if self not in _lfm2_gguf_loader_instances:
#             _lfm2_gguf_loader_instances.append(self)

#     @classmethod
#     def INPUT_TYPES(cls):
#         unet_names = [x for x in folder_paths.get_filename_list(
#             "unet_gguf") if "lfm2-vl" in x.lower() and "mmproj" not in x.lower()]
#         clip_names = [x for x in folder_paths.get_filename_list(
#             "clip_gguf") if "mmproj-lfm2-vl" in x.lower()]
#         return {
#             "required": {
#                 "model_name": (unet_names,),
#                 "vision_model_name": (clip_names,),
#                 "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000}),
#                 "n_ctx": ("INT", {"default": 4096, "min": 2048, "max": 128000}),
#                 "device": (["cuda", "cpu"], {"default": "cuda"}),
#                 "verbose": (["enable", "disable"], {"default": "enable"}),
#             }
#         }

#     RETURN_TYPES = ("LFM2_GGUF_MODEL",)
#     RETURN_NAMES = ("lfm2_gguf_model",)
#     FUNCTION = "load_model"
#     CATEGORY = "VL-Nodes/LFM2-VL"

#     def unload(self):
#         if self.llm is None and self.chat_handler is None:
#             return
#         print("LFM2 GGUF: Unloading model and vision handler.")
#         if self.chat_handler is not None:
#             if self.llm is not None and hasattr(self.llm, 'chat_handler'):
#                 self.llm.chat_handler = None
#             del self.chat_handler
#             self.chat_handler = None
#         if self.llm is not None:
#             del self.llm
#             self.llm = None
#         self.initial_state = None
#         self.cached_params = {}
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def load_model(self, model_name, vision_model_name, n_gpu_layers, n_ctx, device, verbose):
#         current_params = {
#             "model_name": model_name,
#             "vision_model_name": vision_model_name,
#             "n_gpu_layers": n_gpu_layers,
#             "n_ctx": n_ctx,
#             "device": device,
#             "verbose": verbose,
#         }
#         if self.llm is not None and self.cached_params == current_params:
#             print("LFM2 GGUF: Reusing cached model.")
#             return ({"llm": self.llm, "params": self.cached_params, "initial_state": self.initial_state},)

#         if self.llm is not None:
#             print("LFM2 GGUF: Parameters changed, unloading old model.")
#             self.unload()

#         effective_n_gpu_layers = n_gpu_layers
#         if device == 'cpu':
#             effective_n_gpu_layers = 0
#             print("LFM2 GGUF: Loading model to CPU (n_gpu_layers=0).")

#         main_model_path = folder_paths.get_full_path("unet_gguf", model_name)
#         if not main_model_path or not os.path.exists(main_model_path):
#             raise FileNotFoundError(
#                 f"LFM2 GGUF model '{model_name}' not found.")

#         model_dir = os.path.dirname(main_model_path)
#         vision_model_path = os.path.join(model_dir, vision_model_name)
#         if not os.path.exists(vision_model_path):
#             vision_model_path = folder_paths.get_full_path(
#                 "clip_gguf", vision_model_name)
#             if not vision_model_path or not os.path.exists(vision_model_path):
#                 raise FileNotFoundError(
#                     f"LFM2 vision model '{vision_model_name}' not found in model dir or clip_gguf folder.")

#         print("LFM2 GGUF: Loading new model...")
#         # LFM2-VL models use the ChatML format, which we must specify here.
#         self.chat_handler = CustomLlava15ChatHandler(
#             clip_model_path=vision_model_path, verbose=(verbose == "enable"))
#         self.llm = Llama(
#             model_path=main_model_path,
#             chat_handler=self.chat_handler,
#             chat_format="chatml",
#             n_ctx=n_ctx,
#             n_threads=8,
#             n_gpu_layers=effective_n_gpu_layers,
#             verbose=(verbose == "enable"),
#         )

#         try:
#             # Eagerly initialize the multimodal context to catch errors early.
#             print("LFM2 GGUF: Eagerly initializing vision model context...")
#             self.llm.chat_handler._init_mtmd_context(self.llm)
#             print("LFM2 GGUF: Vision model context loaded successfully.")
#             self.cached_params = current_params
#             self.initial_state = self.llm.save_state()
#         except Exception as e:
#             self.unload()  # Unload everything if vision model fails to load
#             error_message = str(e)
#             if "unknown projector type: lfm2" in error_message:
#                 raise RuntimeError(
#                     "Failed to load LFM2 vision model due to 'unknown projector type: lfm2'. "
#                     "This means your version of llama-cpp-python is too old. "
#                     "Please upgrade it in your ComfyUI venv by running: "
#                     "pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir"
#                 ) from e
#             raise RuntimeError(
#                 f"Failed to load the vision model ({vision_model_name}). Please ensure it's a matching pair for the main model. Original error: {e}"
#             ) from e
#         return ({"llm": self.llm, "params": self.cached_params, "initial_state": self.initial_state},)


# class LFM2GGUFImageToText:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "lfm2_gguf_model": ("LFM2_GGUF_MODEL",),
#                 "image": ("IMAGE",),
#                 "system_prompt": ("STRING", {"default": "You are a helpful multimodal assistant by Liquid AI.", "multiline": True}),
#                 "prompt": ("STRING", {"default": "Describe this image.", "multiline": True}),
#                 "max_tokens": ("INT", {"default": 512, "min": -1, "max": 4096}),
#                 "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.1}),
#                 "min_p": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.05}),
#             },
#         }

#     RETURN_TYPES = ("STRING",)
#     RETURN_NAMES = ("text",)
#     FUNCTION = "generate_text"
#     CATEGORY = "VL-Nodes/LFM2-VL"

#     def generate_text(self, lfm2_gguf_model, image, system_prompt, prompt, max_tokens, temperature, min_p, repetition_penalty):
#         llm = lfm2_gguf_model["llm"]
#         initial_state = lfm2_gguf_model["initial_state"]
#         llm.load_state(initial_state)
#         pil_image = tensor2pil(image)

#         with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
#             pil_image.save(tmpfile, "PNG")
#             image_path = tmpfile.name

#         image_uri = "data:image/png;base64," + \
#             base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

#         try:
#             messages = []
#             if system_prompt and system_prompt.strip():
#                 messages.append({"role": "system", "content": system_prompt})

#             messages.append({
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": image_uri}},
#                     {"type": "text", "text": prompt},
#                 ],
#             })

#             response = llm.create_chat_completion(
#                 messages=messages,
#                 max_tokens=max_tokens if max_tokens > 0 else None,
#                 temperature=temperature,
#                 min_p=min_p,
#                 repeat_penalty=repetition_penalty,
#             )
#             description = response['choices'][0]['message']['content']
#         finally:
#             os.remove(image_path)

#         # Extract only the assistant's response
#         assistant_marker = "assistant\n"
#         last_occurrence = description.rfind(assistant_marker)
#         if last_occurrence != -1:
#             final_text = description[last_occurrence +
#                                      len(assistant_marker):].strip()
#             return (final_text,)

#         if "assistant" in description:
#             description = description.split("assistant")[-1].strip()

#         return (description.strip(),)

NODE_CLASS_MAPPINGS = {
    "LFM2TransformerModelLoader": LFM2TransformerModelLoader,
    "LFM2TransformerImageToText": LFM2TransformerImageToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LFM2TransformerModelLoader": "Load LFM2-VL HF Model",
    "LFM2TransformerImageToText": "LFM2-VL HF Image to Text",
}
