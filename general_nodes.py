
import time
import gc
import torch
import requests
from comfy.cli_args import args
import comfy.model_management
from .utils import any_type
from .mimo_nodes import unload_all_mimo_models
from .lfm2_nodes import unload_all_lfm2_hf_models
from .ovisu1_nodes import unload_all_ovisu1_models
from .ovis25_nodes import unload_all_ovis25_models


class MiMoFreeMemoryAPI:
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
                f"MiMoFreeMemoryAPI: Calling ComfyUI API to free memory: {url}")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print("MiMoFreeMemoryAPI: Successfully freed models and execution cache.")
            if self.add_waiting:
                time.sleep(1)
        except Exception as e:
            print(f"MiMoFreeMemoryAPI: Failed to call /free API: {e}")

        return (anything,)


NODE_CLASS_MAPPINGS = {
    "MiMoFreeMemoryAPI": MiMoFreeMemoryAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiMoFreeMemoryAPI": "Free Memory (VL Nodes)",
}
