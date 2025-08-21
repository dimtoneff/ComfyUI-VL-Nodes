import os
import re
import gc
import torch
import tempfile
import base64
import html
from llama_cpp import Llama
import folder_paths
import comfy.model_management
from ..utils import tensor2pil, resize_pil_image, CustomQwen25VLChatHandler, update_folder_names_and_paths

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])

_mimo_loader_instances = []


def unload_all_mimo_models():
    """Unloads all MiMo GGUF models and releases their resources."""
    global _mimo_loader_instances
    if not _mimo_loader_instances:
        return

    print("Unloading all MiMo GGUF models...")
    for loader in _mimo_loader_instances:
        loader.unload()


class MiMoModelLoader:
    def __init__(self):
        global _mimo_loader_instances
        self.llm = None
        self.chat_handler = None
        self.cached_params = {}
        if self not in _mimo_loader_instances:
            _mimo_loader_instances.append(self)

    def __del__(self):
        self.unload()
        if self in _mimo_loader_instances:
            _mimo_loader_instances.remove(self)

    @classmethod
    def INPUT_TYPES(cls):
        unet_names = [
            x for x in folder_paths.get_filename_list("unet_gguf") if "mimo-vl" in x.lower() and "mmproj" not in x.lower()
        ]
        clip_names = [
            x for x in folder_paths.get_filename_list("clip_gguf") if "mmproj-mimo" in x.lower()
        ]
        return {
            "required": {
                "model_name": (unet_names,),
                "chat_handler": (clip_names,),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1, "tooltip": "Number of layers to offload to GPU. -1 for all."}),
                "n_ctx": ("INT", {"default": 4096, "min": 2048, "max": 128000, "step": 1024, "tooltip": "Context length. Lower this if you run out of VRAM."}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "verbose": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("MIMO_MODEL",)
    RETURN_NAMES = ("mimo_model",)
    FUNCTION = "load_model"
    CATEGORY = "VL-Nodes/MiMo"

    def unload(self):
        """Unloads the GGUF model and releases associated resources."""
        if self.llm is None and self.chat_handler is None:
            return

        print("MiMo GGUF: Unloading model and vision handler.")

        # The order is important: The chat_handler's mtmd_ctx holds a reference
        # to the C-level llama_model, so it must be freed before the Llama object.

        # 1. Unload the vision model and mtmd_ctx from the chat handler.
        if self.chat_handler is not None:
            # The Llama object holds a reference to the chat_handler.
            # Break the reference cycle to allow the chat_handler to be garbage collected.
            if self.llm is not None and hasattr(self.llm, 'chat_handler'):
                self.llm.chat_handler = None

            # Deleting the chat_handler will trigger its __del__ method, which calls
            # close() and releases the C-level mtmd_ctx resources.
            print(
                "MiMo GGUF: Unloading vision model (CLIP) and mtmd context via custom handler.")
            del self.chat_handler
            self.chat_handler = None

        # 2. Unload the main GGUF model by deleting the Llama object.
        if self.llm is not None:
            del self.llm
            self.llm = None

        # 3. Reset cache and force garbage collection.
        self.cached_params = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_name, chat_handler, n_gpu_layers, n_ctx, verbose, device):
        current_params = {
            "model_name": model_name,
            "chat_handler": chat_handler,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "verbose": verbose,
            "device": device,
        }

        if self.llm is not None and self.cached_params == current_params:
            print("MiMo GGUF: Reusing cached model.")
            model_data = {"llm": self.llm, "params": self.cached_params}
            return (model_data,)

        if self.llm is not None:
            print("MiMo GGUF: Parameters changed, unloading old model.")
            self.unload()

        effective_n_gpu_layers = n_gpu_layers
        if device == 'cpu':
            effective_n_gpu_layers = 0
            print("MiMo GGUF: Loading model to CPU (n_gpu_layers=0).")

        # Use get_full_path to find the main model, which can be in a subfolder
        main_model_path = folder_paths.get_full_path("unet_gguf", model_name)
        # The vision model must be in the same directory as the main model
        model_dir = os.path.dirname(
            main_model_path) if main_model_path else folder_paths.get_folder_paths("unet_gguf")[0]

        vision_model_name = chat_handler  # this is the filename from input
        vision_model_path = os.path.join(model_dir, vision_model_name)

        if not main_model_path or not os.path.exists(main_model_path):
            raise FileNotFoundError(
                f"MiMo model '{model_name}' not found. Please download it or select 'yes' to download.")

        if not os.path.exists(vision_model_path):
            raise FileNotFoundError(
                f"MiMo vision model '{vision_model_name}' not found in '{model_dir}'. It is required for image processing and must be in the same directory as the main model.")

        print("MiMo GGUF: Loading new model...")
        try:
            # For this model, we must use the Qwen25VLChatHandler which handles the vision part. Due to a memory leak we are using our custom handler.
            self.chat_handler = CustomQwen25VLChatHandler(
                clip_model_path=vision_model_path, verbose=(verbose == "enable"))

            self.llm = Llama(
                model_path=main_model_path,
                chat_handler=self.chat_handler,
                n_ctx=n_ctx,
                n_threads=8,
                n_gpu_layers=effective_n_gpu_layers,
                verbose=(verbose == "enable"),
            )
            self.cached_params = current_params

            model_data = {"llm": self.llm, "params": self.cached_params}
            return (model_data,)
        except Exception as e:
            print(f"MiMo GGUF: Error loading model: {e}")
            self.unload()  # Attempt to clean up resources
            raise e


class MiMoImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mimo_model": ("MIMO_MODEL",),
                "image": ("IMAGE",),
                "resize_image": ("BOOLEAN", {"default": True, "label_on": "Resize Enabled", "label_off": "Resize Disabled"}),
                "system_prompt": ("STRING", {"default": "You are an assistant who perfectly describes images. " + "Fully describe the image as a prompt for Stable Diffusion text2image SDXL model. If you see nsfw stuff describe it too. Do not omit anything! Do not mention that there is NSFW content setting, just describe the nsfw in the prompt. " +
                                             "!Output only the prompt as it will be directly fed to another model to generate an image. Be clear and use short sentences to describe the image!", "multiline": True}),
                "prompt": ("STRING", {"default": "What it is on this image?", "multiline": True}),
                "max_tokens": ("INT", {"default": 128000, "min": 4096, "max": 128000, "step": 1, "tooltip": "Max tokens for the response. -1 means unlimited (up to context size)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("Prompt", "Thinking")
    FUNCTION = "generate_description"
    CATEGORY = "VL-Nodes/MiMo"

    def generate_description(self, mimo_model, image, resize_image, system_prompt, prompt, max_tokens):
        pil_image = tensor2pil(image)
        original_llm = mimo_model["llm"]
        params = mimo_model["params"]

        is_on_cpu = params.get("device", "cuda") == "cpu"
        gpu_available = comfy.model_management.get_torch_device().type == 'cuda'

        llm_for_inference = original_llm
        temp_gpu_llm = None
        image_path = None
        description = ""
        thinking_text = ""
        prompt_text = ""

        try:
            if is_on_cpu and gpu_available:
                print(
                    "MiMo GGUF: Temporarily loading a copy of the model to GPU for inference.")
                main_model_path = folder_paths.get_full_path(
                    "unet_gguf", params["model_name"])
                model_dir = os.path.dirname(
                    main_model_path) if main_model_path else folder_paths.get_folder_paths("unet_gguf")[0]
                vision_model_path = os.path.join(
                    model_dir, params["chat_handler"])
                chat_handler_gpu = CustomQwen25VLChatHandler(
                    clip_model_path=vision_model_path, verbose=(params["verbose"] == "enable"))
                temp_gpu_llm = Llama(
                    model_path=main_model_path,
                    chat_handler=chat_handler_gpu,
                    n_ctx=params["n_ctx"],
                    n_threads=8,
                    n_gpu_layers=params["n_gpu_layers"],
                    verbose=(params["verbose"] == "enable"),
                )
                llm_for_inference = temp_gpu_llm

            if resize_image:
                pil_image = resize_pil_image(
                    pil_image, target_size=512, node_name="MiMoImageToText")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                pil_image.save(tmpfile, "PNG")
                image_path = tmpfile.name

            image_uri = "data:image/png;base64," + \
                base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

            response = llm_for_inference.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            description = response['choices'][0]['message']['content']
            description = html.unescape(description)

        except Exception as e:
            print(f"MiMo GGUF: Error during image to text generation: {e}")
            thinking_text = f"Error during generation: {e}"

        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            if temp_gpu_llm is not None:
                print("MiMo GGUF: Unloading temporary GPU model.")
                if hasattr(temp_gpu_llm, 'chat_handler') and temp_gpu_llm.chat_handler is not None:
                    del temp_gpu_llm.chat_handler
                del temp_gpu_llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not thinking_text:  # No error occurred
            prompt_text = description
            think_match = re.search(
                r'<think>(.*?)</think>', description if description is not None else "", re.DOTALL)
            if think_match:
                thinking_text = think_match.group(1).strip()
                prompt_text = (description or "").replace(
                    think_match.group(0), '').strip()
        else:  # Error occurred
            prompt_text = ""  # Return empty prompt text

        return ((prompt_text or "").strip('"'), thinking_text,)


NODE_CLASS_MAPPINGS = {
    "MiMoModelLoader": MiMoModelLoader,
    "MiMoImageToText": MiMoImageToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiMoModelLoader": "Load MiMo GGUF Model",
    "MiMoImageToText": "MiMo Image to Text",
}
