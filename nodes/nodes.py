import hashlib
import json
import os
import shutil
import sys
import zipfile
from io import BytesIO

import folder_paths
import node_helpers
import numpy as np
import torch
from comfy_extras.nodes_audio import SaveAudio
from PIL import Image, ImageOps, ImageSequence, PngImagePlugin
from PIL.PngImagePlugin import PngInfo

from .monkeypatch import set_bentoml_output


# AnyType class hijacks the isinstance, issubclass, bool, str, jsonserializable, eq, ne methods to always return True
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")  # when a != b is called, it will always return False


class OutputFile:
    COLOR = (142, 36, 170)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "", "forceInput": True}),
                "filename_prefix": ("STRING", {"default": "cpack_output_"}),
            },
        }

    RETURN_TYPES = ()
    CATEGORY = "ComfyPack/output"
    CPACK_NODE = True
    FUNCTION = "save"
    DESCRIPTION = "Save the input data for comfy-pack output"

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
                "filename_prefix": ("STRING", {"default": "cpack_output_"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    CPACK_NODE = True
    OUTPUT_NODE = True

    CATEGORY = "ComfyPack/output"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(
        self, images, filename_prefix="cpack_output_", prompt=None, extra_pnginfo=None
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


class OutputImageWithStringTxt:
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
                "filename_prefix": ("STRING", {"default": "cpack_output_"}),
                "text": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    CPACK_NODE = True
    OUTPUT_NODE = True

    CATEGORY = "ComfyPack/output"
    DESCRIPTION = (
        "Saves the input images (and optional text) to your ComfyUI output directory."
    )

    def save_images(
        self,
        images,
        filename_prefix="cpack_output_",
        text="",
        prompt=None,
        extra_pnginfo=None,
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )

        base_counter = counter  # use for name zip
        zip_filename = f"{filename}_batch_{base_counter:05}.zip"
        zip_path = os.path.join(full_output_folder, zip_filename)

        # create ZIP file
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for batch_number, image in enumerate(images):
                # temp store img to RAM
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                # write meta data
                metadata = PngImagePlugin.PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                # write img file to RAM buffer
                img_buffer = BytesIO()
                img.save(
                    img_buffer,
                    format="PNG",
                    pnginfo=metadata,
                    compress_level=self.compress_level,
                )
                img_buffer.seek(0)

                # write img into ZIP file
                image_filename = f"image_{batch_number:05}.png"
                zipf.writestr(image_filename, img_buffer.read())

                # write txt into ZIP file
                text_filename = f"text_{batch_number:05}.txt"
                zipf.writestr(text_filename, text)

        # return zip as output
        out = [{"filename": zip_filename, "subfolder": subfolder, "type": "zip"}]
        return {
            "ui": {
                "zip": out,
            }
        }


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

    CATEGORY = "ComfyPack/input"
    CPACK_NODE = True
    RETURN_TYPES = ("IMAGE", "MASK")
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


class FileInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (anytype,)
    RETURN_NAMES = ("path",)
    FUNCTION = "identity"
    CPACK_NODE = True
    CATEGORY = "ComfyPack/input"

    def identity(self, path):
        return (path,)

    @classmethod
    def VALIDATE_INPUTS(s, path):
        set_bentoml_output([(path,)])
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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "identity"
    CPACK_NODE = True
    CATEGORY = "ComfyPack/input"

    def identity(self, value):
        return (value,)

    @classmethod
    def VALIDATE_INPUTS(s, value):
        set_bentoml_output([(value,)])
        return True


class IntInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("INT", {"default": 0}),
            },
            "optional": {
                "min": ("INT", {"default": -sys.maxsize}),
                "max": ("INT", {"default": sys.maxsize}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "identity"
    CPACK_NODE = True
    CATEGORY = "ComfyPack/input"

    def identity(self, value, min=None, max=None):
        return (value,)

    @classmethod
    def VALIDATE_INPUTS(s, value, min=None, max=None):
        if min is not None and max is not None and min > max:
            return f"Value must be greater than or equal to {min}"
        set_bentoml_output([(value,)])
        return True


class AnyInput:
    COLOR = (142, 36, 170)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("*", {"default": ""}),
            }
        }

    RETURN_TYPES = (anytype,)
    RENAME = ("value",)
    FUNCTION = "identity"
    CPACK_NODE = True
    CATEGORY = "ComfyPack/input"

    def identity(self, input):
        return (input,)

    @classmethod
    def VALIDATE_INPUTS(s, input):
        set_bentoml_output([(input,)])
        return True


class OutputAudio(SaveAudio):
    CPACK_NODE = True
    CATEGORY = "ComfyPack/output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "cpack_output_"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }


NODE_CLASS_MAPPINGS = {
    "CPackOutputFile": OutputFile,
    "CPackOutputImage": OutputImage,
    "CPackOutputAudio": OutputAudio,
    "CPackOutputZip": OutputImageWithStringTxt,
    "CPackInputImage": ImageInput,
    "CPackInputString": StringInput,
    "CPackInputInt": IntInput,
    "CPackInputFile": FileInput,
    "CPackInputAny": AnyInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CPackInputImage": "Image Input",
    "CPackInputString": "String Input",
    "CPackInputInt": "Int Input",
    "CPackInputFile": "File Input",
    "CPackInputAny": "Any Input",
    "CPackOutputImage": "Image Output",
    "CPackOutputAudio": "Audio Output",
    "CPackOutputFile": "File Output",
    "CPackOutputZip": "Zip Output(img + txt file)",
}
