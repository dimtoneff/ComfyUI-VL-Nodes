import base64
import gc
import html
import os
import re
import tempfile

import torch
from llama_cpp import Llama

import comfy.model_management
import folder_paths

from ..utils import CustomQwen25VLChatHandler, CustomLlava15ChatHandler, resize_pil_image, tensor2pil, update_folder_names_and_paths

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])

_gguf_loader_instances = []


def unload_all_gguf_models():
    """Unloads all GGUF models and releases their resources."""
    global _gguf_loader_instances
    if not _gguf_loader_instances:
        return

    print("Unloading all GGUF models...")
    for loader in _gguf_loader_instances:
        loader.unload()


class GGUF_VLM_ModelLoader:
    def __init__(self):
        global _gguf_loader_instances
        self.llm = None
        self.chat_handler = None
        self.cached_params = {}
        if self not in _gguf_loader_instances:
            _gguf_loader_instances.append(self)

    def __del__(self):
        self.unload()
        if self in _gguf_loader_instances:
            _gguf_loader_instances.remove(self)

    @classmethod
    def INPUT_TYPES(cls):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf") if ".gguf" in x.lower() and ("mimo" in x.lower() or "minicpm" in x.lower() or "internvl3_5" in x.lower()) and "mmproj" not in x.lower()]
        clip_names = [x for x in folder_paths.get_filename_list("clip_gguf") if ("mimo" in x.lower() or "minicpm" in x.lower() or "internvl3_5" in x.lower()) and "mmproj" in x.lower()]
        return {
            "required": {
                "model_name": (unet_names,),
                "vision_model_name": (clip_names,),
                "chat_handler_type": (["Qwen-VL", "LLaVA-1.5"], {"default": "LLaVA-1.5"}, {"tooltip": "Use Qwen-VL for MiMo-VL, InternVL3.5 models, LLaVA-1.5 for MiniCPM"}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1, "tooltip": "Number of layers to offload to GPU. -1 for all."}),
                "n_ctx": ("INT", {"default": 4096, "min": 2048, "max": 128000, "step": 1024, "tooltip": "Context length. Lower this if you run out of VRAM."}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "verbose": (["enable", "disable"], {"default": "enable"}),
            },
            "optional": {
                "chat_template": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("GGUF_MODEL",)
    RETURN_NAMES = ("gguf_model",)
    FUNCTION = "load_model"
    CATEGORY = "VL-Nodes/GGUF"

    def unload(self):
        """Unloads the GGUF model and releases associated resources."""
        if self.llm is None and self.chat_handler is None:
            return

        print("GGUF VLM: Unloading model and vision handler.")

        # The order is important: The chat_handler's mtmd_ctx holds a reference
        # to the C-level llama_model, so it must be freed before the Llama object.

        # 1. Unload the vision model and mtmd_ctx from the chat handler.
        if self.chat_handler is not None:
            # The Llama object holds a reference to the chat_handler.
            # Break the reference cycle to allow the chat_handler to be garbage collected.
            if self.llm is not None and hasattr(self.llm, "chat_handler"):
                self.llm.chat_handler = None

            # Deleting the chat_handler will trigger its __del__ method, which calls
            # close() and releases the C-level mtmd_ctx resources.
            print("GGUF VLM: Unloading vision model (CLIP) and mtmd context via custom handler.")
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

    def load_model(self, model_name, vision_model_name, chat_handler_type, n_gpu_layers, n_ctx, verbose, device, chat_template=""):
        current_params = {
            "model_name": model_name,
            "vision_model_name": vision_model_name,
            "chat_handler_type": chat_handler_type,
            "n_gpu_layers": n_gpu_layers,
            "n_ctx": n_ctx,
            "verbose": verbose,
            "device": device,
            "chat_template": chat_template,
        }

        if self.llm is not None and self.cached_params == current_params:
            print("GGUF VLM: Reusing cached model.")
            model_data = {"llm": self.llm, "params": self.cached_params}
            return (model_data,)

        if self.llm is not None:
            print("GGUF VLM: Parameters changed, unloading old model.")
            self.unload()

        effective_n_gpu_layers = n_gpu_layers
        if device == "cpu":
            effective_n_gpu_layers = 0
            print("GGUF VLM: Loading model to CPU (n_gpu_layers=0).")

        # Use get_full_path to find the main model, which can be in a subfolder
        main_model_path = folder_paths.get_full_path("unet_gguf", model_name)
        # The vision model must be in the same directory as the main model
        model_dir = os.path.dirname(main_model_path) if main_model_path else folder_paths.get_folder_paths("unet_gguf")[0]

        vision_model_path = os.path.join(model_dir, vision_model_name)

        if not main_model_path or not os.path.exists(main_model_path):
            raise FileNotFoundError(f"GGUF model '{model_name}' not found. Please download it or select 'yes' to download.")

        if not os.path.exists(vision_model_path):
            raise FileNotFoundError(f"GGUF vision model '{vision_model_name}' not found in '{model_dir}'. It is required for image processing and must be in the same directory as the main model.")

        print("GGUF VLM: Loading new model...")
        try:
            handler_kwargs = {"clip_model_path": vision_model_path, "verbose": (verbose == "enable")}

            if chat_handler_type == "Qwen-VL":
                print("GGUF VLM: Using Qwen-VL chat handler.")
                self.chat_handler = CustomQwen25VLChatHandler(**handler_kwargs)
            elif chat_handler_type == "LLaVA-1.5":
                print("GGUF VLM: Using LLaVA-1.5 chat handler.")
                self.chat_handler = CustomLlava15ChatHandler(**handler_kwargs)
            else:
                raise ValueError(f"Unsupported chat handler type: {chat_handler_type}")

            if chat_template and chat_template.strip():
                print("GGUF VLM: Applying custom chat template.")
                self.chat_handler.chat_template = chat_template

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
            print(f"GGUF VLM: Error loading model: {e}")
            self.unload()  # Attempt to clean up resources
            raise e


class GGUF_VLM_ImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_model": ("GGUF_MODEL",),
                "image": ("IMAGE",),
                "resize_image": ("BOOLEAN", {"default": True, "label_on": "Resize Enabled", "label_off": "Resize Disabled"}),
                "enable_thinking": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are an assistant who perfectly describes images. "
                        + "Fully describe the image as a prompt for Stable Diffusion text2image SDXL model. If you see nsfw stuff describe it too. Do not omit anything! Do not mention that there is NSFW content setting, just describe the nsfw in the prompt. "
                        + "!Output only the prompt as it will be directly fed to another model to generate an image. Be clear and use short sentences to describe the image!",
                        "multiline": True,
                    },
                ),
                "prompt": ("STRING", {"default": "What it is on this image?", "multiline": True}),
                "special_captioning_token": ("STRING", {"default": "", "multiline": False}),
                "seed": ("INT", {"default": 69, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 128000, "min": 4096, "max": 128000, "step": 1, "tooltip": "Max tokens for the response. -1 means unlimited (up to context size)."}),
            },
            "optional": {
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "thinking", "text_list", "thinking_list")
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "generate_description"
    CATEGORY = "VL-Nodes/GGUF"

    def generate_description(self, gguf_model, image, resize_image, enable_thinking, system_prompt, special_captioning_token, seed, prompt, max_tokens, temperature, top_k, top_p, min_p, repeat_penalty):
        pil_images = tensor2pil(image)
        batch_size = len(pil_images)
        original_llm = gguf_model["llm"]
        params = gguf_model["params"]

        is_on_cpu = params.get("device", "cuda") == "cpu"
        gpu_available = comfy.model_management.get_torch_device().type == "cuda"

        llm_for_inference = original_llm
        temp_gpu_llm = None

        prompts = []
        thinkings = []

        try:
            if is_on_cpu and gpu_available:
                print("GGUF VLM: Temporarily loading a copy of the model to GPU for inference.")
                main_model_path = folder_paths.get_full_path("unet_gguf", params["model_name"])
                model_dir = os.path.dirname(main_model_path) if main_model_path else folder_paths.get_folder_paths("unet_gguf")[0]
                vision_model_path = os.path.join(model_dir, params["vision_model_name"])

                chat_handler_type = params.get("chat_handler_type", "Qwen-VL")
                chat_template = params.get("chat_template", "")

                handler_kwargs = {"clip_model_path": vision_model_path, "verbose": (params["verbose"] == "enable")}

                if chat_handler_type == "Qwen-VL":
                    chat_handler_gpu = CustomQwen25VLChatHandler(**handler_kwargs)
                elif chat_handler_type == "LLaVA-1.5":
                    chat_handler_gpu = CustomLlava15ChatHandler(**handler_kwargs)
                else:
                    raise ValueError(f"Unsupported chat handler type: {chat_handler_type}")

                if chat_template and chat_template.strip():
                    chat_handler_gpu.chat_template = chat_template

                temp_gpu_llm = Llama(
                    model_path=main_model_path,
                    chat_handler=chat_handler_gpu,
                    n_ctx=params["n_ctx"],
                    n_threads=8,
                    n_gpu_layers=params["n_gpu_layers"],
                    verbose=(params["verbose"] == "enable"),
                )
                llm_for_inference = temp_gpu_llm
                print(f"GGUF VLM Caption: Batch size: {batch_size}")

            for i in range(batch_size):
                pil_image = pil_images[i]
                image_path = None
                description = ""
                thinking_text = ""
                prompt_text = ""
                print(f"GGUF VLM Caption: Processing image {i + 1}/{batch_size}")

                try:
                    if resize_image:
                        pil_image = resize_pil_image(pil_image, target_size=512, node_name="GGUF_VLM_ImageToText")

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        pil_image.save(tmpfile, "PNG")
                        image_path = tmpfile.name

                    image_uri = "data:image/png;base64," + base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

                    current_system_prompt = f"/think {system_prompt}" if enable_thinking else f"/no_think {system_prompt}"

                    response = llm_for_inference.create_chat_completion(
                        messages=[
                            {"role": "system", "content": current_system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": image_uri}},
                                    {"type": "text", "text": prompt},
                                ],
                            },
                        ],
                        max_tokens=max_tokens,
                        seed=seed,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        min_p=min_p,
                        repeat_penalty=repeat_penalty,
                    )
                    print(f"GGUF VLM Raw Response: {response}")
                    description = response["choices"][0]["message"]["content"]
                    description = html.unescape(description)

                except Exception as e:
                    print(f"GGUF VLM: Error during image to text generation: {e}")
                    thinking_text = f"Error during generation: {e}"

                finally:
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)

                if not thinking_text:  # No error occurred
                    prompt_text = description
                    think_match = re.search(r"<think>(.*?)</think>", description if description is not None else "", re.DOTALL)
                    if think_match:
                        thinking_text = think_match.group(1).strip()
                        prompt_text = (description or "").replace(think_match.group(0), "").strip()
                else:  # Error occurred
                    prompt_text = ""  # Return empty prompt text

                if special_captioning_token and special_captioning_token.strip():
                    prompt_text = f"{special_captioning_token.strip()}, {prompt_text}"

                prompt_text = " ".join(prompt_text.split())
                prompt_text = prompt_text.lstrip(": ")
                prompts.append((prompt_text or "").strip('"'))
                thinkings.append(thinking_text)

        finally:
            if temp_gpu_llm is not None:
                print("GGUF VLM: Unloading temporary GPU model.")
                if hasattr(temp_gpu_llm, "chat_handler") and temp_gpu_llm.chat_handler is not None:
                    del temp_gpu_llm.chat_handler
                del temp_gpu_llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return ("\n\n=============================\n\n".join(prompts), "\n\n=============================\n\n".join(thinkings), tuple(prompts), tuple(thinkings))


NODE_CLASS_MAPPINGS = {
    "GGUF_VLM_ModelLoader": GGUF_VLM_ModelLoader,
    "GGUF_VLM_ImageToText": GGUF_VLM_ImageToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GGUF_VLM_ModelLoader": "Load GGUF VLM Model",
    "GGUF_VLM_ImageToText": "GGUF VLM Image to Text",
}
