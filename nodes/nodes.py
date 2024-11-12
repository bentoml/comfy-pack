import hashlib
import json
import os
import shutil

import folder_paths
import node_helpers
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

from .monkeypatch import set_bentoml_output


# AnyType class hijacks the isinstance, issubclass, bool, str, jsonserializable, eq, ne methods to always return True
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")  # when a != b is called, it will always return False


class OutputPath:
    COLOR = (142, 36, 170)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "", "forceInput": True}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_IDL_"}),
            },
        }

    RETURN_TYPES = ()
    CATEGORY = "ComfyUI-IDL/output"
    BENTOML_NODE = True
    FUNCTION = "save"
    DESCRIPTION = "Save the input data for IDL output"

    def save(self, filename, filename_prefix):
        if not filename_prefix:
            return ()

        subfolder, prefix = os.path.split(filename_prefix)
        if subfolder:
            os.makedirs(subfolder, exist_ok=True)
        else:
            subfolder = os.path.dirname(filename)
        basename = os.path.basename(filename)
        new_filename = os.path.join(subfolder, f"{prefix}{basename}")
        shutil.copy2(filename, new_filename)
        return ()


def get_save_image_path(
    filename_prefix: str,
    output_dir: str,
    image_width=0,
    image_height=0,
) -> tuple[str, str, int, str, str]:
    def map_filename(filename: str) -> tuple[int, str]:
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except:
            digits = 0
        return digits, prefix

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)

    try:
        counter = (
            max(
                filter(
                    lambda a: os.path.normcase(a[1][:-1]) == os.path.normcase(filename)
                    and a[1][-1] == "_",
                    map(map_filename, os.listdir(full_output_folder)),
                )
            )[0]
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


class OutputImage:
    COLOR = (142, 36, 170)

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_IDL_"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    BENTOML_NODE = True
    OUTPUT_NODE = True

    CATEGORY = "ComfyUI-IDL/output"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(
        self, images, filename_prefix="ComfyUI_IDL_", prompt=None, extra_pnginfo=None
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


class ImageInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {"image": (sorted(files), {"image_upload": True})},
        }

    CATEGORY = "ComfyUI-IDL/input"
    BENTOML_NODE = True
    RETURN_TYPES = ("IMAGE", "MASK")
    RETUEN_NAMES = ("image", "mask")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class StringInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("STRING", {"default": ""}),
            }
        }

    RETUEN_NAMES = ("string",)
    RETURN_TYPES = (anytype,)
    BENTOML_NODE = True
    FUNCTION = "string_input"
    CATEGORY = "ComfyUI-IDL/input"

    def string_input(self, value):
        return (value,)

    @classmethod
    def VALIDATE_INPUTS(s, value):
        set_bentoml_output([(value,)])
        return True


class SelectionInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected": ("STRING", {"default": ""}),
                "options": ("STRING", {"multiline": True, "default": ""}),
                "start_index": (["0", "1"], {"default": "0"}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "INT",
    )
    FUNCTION = "process"
    CATEGORY = "ComfyUI-IDL/input"

    @classmethod
    def process(cls, selected, options, start_index):
        options_list = [opt.strip() for opt in options.split("\n") if opt.strip()]

        if not options_list:
            return ("", -1)

        assert selected in options_list

        index = options_list.index(selected)
        return_index = index + (1 if start_index == "1" else 0)

        return (
            selected,
            return_index,
        )

    @classmethod
    def VALIDATE_INPUTS(cls, selected, options, start_index):
        try:
            set_bentoml_output([cls.process(selected, options, start_index)])
            return True
        except AssertionError:
            return False

    @classmethod
    def IS_CHANGED(cls, selected, options, start_index):
        options_list = [opt.strip() for opt in options.split("\n") if opt.strip()]
        return hash((selected, tuple(options_list), start_index))


class IntegerInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0}),
            }
        }

    RETUEN_NAMES = ("int",)
    RETURN_TYPES = (anytype,)
    FUNCTION = "identity"
    BENTOML_NODE = True
    CATEGORY = "ComfyUI-IDL/input"

    def identity(self, value):
        return (value,)

    @classmethod
    def VALIDATE_INPUTS(s, value):
        set_bentoml_output([(value,)])
        return True


class FloatInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0}),
            }
        }

    RETUEN_NAMES = ("float",)
    RETURN_TYPES = (anytype,)
    FUNCTION = "identity"
    BENTOML_NODE = True
    CATEGORY = "ComfyUI-IDL/input"

    def identity(self, value):
        return (value,)

    @classmethod
    def VALIDATE_INPUTS(s, value):
        set_bentoml_output([(value,)])
        return True


class BooleanInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
            }
        }

    RETUEN_NAMES = ("bool",)
    RETURN_TYPES = (anytype,)
    FUNCTION = "identity"
    CATEGORY = "ComfyUI-IDL/input"
    BENTOML_NODE = True

    def identity(self, value):
        return (value,)

    @classmethod
    def VALIDATE_INPUTS(s, value):
        set_bentoml_output([(value,)])
        return True


class PathInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (anytype,)
    RETUEN_NAMES = ("path",)
    FUNCTION = "identity"
    BENTOML_NODE = True
    CATEGORY = "ComfyUI-IDL/input"

    def identity(self, path):
        return (path,)

    @classmethod
    def VALIDATE_INPUTS(s, path):
        set_bentoml_output([(path,)])
        return True


NODE_CLASS_MAPPINGS = {
    "BentoOutputPath": OutputPath,
    "BentoOutputImage": OutputImage,
    "BentoInputImage": ImageInput,
    "BentoInputString": StringInput,
    "BentoInputSelection": SelectionInput,
    "BentoInputInteger": IntegerInput,
    "BentoInputFloat": FloatInput,
    "BentoInputBoolean": BooleanInput,
    "BentoInputPath": PathInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BentoInputImage": "Image Input",
    "BentoInputPath": "Path Input",
    "BentoInputInteger": "Integer Input",
    "BentoInputFloat": "Float Input",
    "BentoInputBoolean": "Boolean Input",
    "BentoInputString": "String Input",
    "BentoInputSelection": "Selection Input",
    "BentoOutputImage": "Image Output",
    "BentoOutputPath": "Path Output",
}
