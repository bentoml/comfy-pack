import glob
import hashlib
import io
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


class OutputZip:
    CATEGORY = "ComfyPack/output"

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ()
    FUNCTION = "null_op"

    def null_op(self):
        return ()


class OutputAudio:
    CPACK_NODE = True
    CATEGORY = "ComfyPack/output"
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    def save_audio(
        self,
        audio,
        filename_prefix,
        format="flac",
        prompt=None,
        extra_pnginfo=None,
        quality="128k",
    ):
        import av
        import torchaudio

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(filename_prefix, self.output_dir)
        )
        results = []

        # Prepare metadata dictionary
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

        # Opus supported sample rates
        OPUS_RATES = [8000, 12000, 16000, 24000, 48000]

        for batch_number, waveform in enumerate(audio["waveform"].cpu()):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}.{format}"
            output_path = os.path.join(full_output_folder, file)

            # Use original sample rate initially
            sample_rate = audio["sample_rate"]

            # Handle Opus sample rate requirements
            if format == "opus":
                if sample_rate > 48000:
                    sample_rate = 48000
                elif sample_rate not in OPUS_RATES:
                    # Find the next highest supported rate
                    for rate in sorted(OPUS_RATES):
                        if rate > sample_rate:
                            sample_rate = rate
                            break
                    if sample_rate not in OPUS_RATES:  # Fallback if still not supported
                        sample_rate = 48000

                # Resample if necessary
                if sample_rate != audio["sample_rate"]:
                    waveform = torchaudio.functional.resample(
                        waveform, audio["sample_rate"], sample_rate
                    )

            # Create in-memory WAV buffer
            wav_buffer = io.BytesIO()
            torchaudio.save(wav_buffer, waveform, sample_rate, format="WAV")
            wav_buffer.seek(0)  # Rewind for reading

            # Use PyAV to convert and add metadata
            input_container = av.open(wav_buffer)

            # Create output with specified format
            output_buffer = io.BytesIO()
            output_container = av.open(output_buffer, mode="w", format=format)

            # Set metadata on the container
            for key, value in metadata.items():
                output_container.metadata[key] = value

            # Set up the output stream with appropriate properties
            input_container.streams.audio[0]
            if format == "opus":
                out_stream = output_container.add_stream("libopus", rate=sample_rate)
                if quality == "64k":
                    out_stream.bit_rate = 64000
                elif quality == "96k":
                    out_stream.bit_rate = 96000
                elif quality == "128k":
                    out_stream.bit_rate = 128000
                elif quality == "192k":
                    out_stream.bit_rate = 192000
                elif quality == "320k":
                    out_stream.bit_rate = 320000
            elif format == "mp3":
                out_stream = output_container.add_stream("libmp3lame", rate=sample_rate)
                if quality == "V0":
                    # TODO i would really love to support V3 and V5 but there doesn't seem to be a way to set the qscale level, the property below is a bool
                    out_stream.codec_context.qscale = 1
                elif quality == "128k":
                    out_stream.bit_rate = 128000
                elif quality == "320k":
                    out_stream.bit_rate = 320000
            else:  # format == "flac":
                out_stream = output_container.add_stream("flac", rate=sample_rate)

            # Copy frames from input to output
            for frame in input_container.decode(audio=0):
                frame.pts = None  # Let PyAV handle timestamps
                output_container.mux(out_stream.encode(frame))

            # Flush encoder
            output_container.mux(out_stream.encode(None))

            # Close containers
            output_container.close()
            input_container.close()

            # Write the output to file
            output_buffer.seek(0)
            with open(output_path, "wb") as f:
                f.write(output_buffer.getbuffer())

            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"audio": results}}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/comfypack"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }


class OutputVideo:
    CPACK_NODE = True
    CATEGORY = "ComfyPack/output"

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    DESCRIPTION = "Saves the input video to your ComfyUI output directory."
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "video/comfypack",
                        "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                    },
                ),
                "format": (
                    ["auto", "mp4"],
                    {"default": "auto", "tooltip": "The format to save the video as."},
                ),
                "codec": (
                    ["auto", "h264"],
                    {"default": "auto", "tooltip": "The codec to use for the video."},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save_video(
        self,
        video,
        filename_prefix,
        format,
        codec,
        prompt=None,
        extra_pnginfo=None,
    ):
        filename_prefix += self.prefix_append
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(filename_prefix, self.output_dir, width, height)
        )
        results = list()
        saved_metadata = None
        metadata = {}
        if extra_pnginfo is not None:
            metadata.update(extra_pnginfo)
        if prompt is not None:
            metadata["prompt"] = prompt
        if len(metadata) > 0:
            saved_metadata = metadata
        file = f"{filename}_{counter:05}.mp4"
        video.save_to(
            os.path.join(full_output_folder, file),
            format=format,
            codec=codec,
            metadata=saved_metadata,
        )

        results.append({"filename": file, "subfolder": subfolder, "type": self.type})
        counter += 1

        return {"ui": {"images": results, "animated": (True,)}}


class OutputTextFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "cpack_output_"}),
            },
            "optional": {
                "file_extension": ("STRING", {"default": ".txt"}),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_text_file"
    CATEGORY = "ComfyPack/output"
    CPACK_NODE = True

    def save_text_file(
        self, text: str, filename_prefix: str, file_extension: str = ".txt"
    ):
        subfolder, filename_prefix = os.path.split(os.path.normpath(filename_prefix))
        output_dir = folder_paths.get_output_directory()
        full_output_folder = os.path.join(output_dir, subfolder)

        full_output_filename = self.get_output_filename(
            full_output_folder, filename_prefix, file_extension
        )
        with open(full_output_filename, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return (text, {"ui": {"string": text}})

    @staticmethod
    def get_output_filename(folder: str, prefix: str, extension: str) -> str:
        matched_files = [
            os.path.basename(f)[len(prefix) + 1 : -len(extension)]
            for f in glob.glob(os.path.join(folder, f"{prefix}_*{extension}"))
        ]
        print("MATCHING", matched_files)
        max_count = max(
            (int(name) for name in matched_files if name.isdigit()), default=0
        )
        return os.path.join(folder, f"{prefix}_{max_count + 1:04d}{extension}")


NODE_CLASS_MAPPINGS = {
    "CPackOutputFile": OutputFile,
    "CPackOutputImage": OutputImage,
    "CPackOutputAudio": OutputAudio,
    "CPackOutputVideo": OutputVideo,
    "CPackOutputZip": OutputImageWithStringTxt,
    "CPackOutputZipSwitch": OutputZip,
    "CPackInputImage": ImageInput,
    "CPackInputString": StringInput,
    "CPackInputInt": IntInput,
    "CPackInputFile": FileInput,
    "CPackInputAny": AnyInput,
    "CPackOutputTextFile": OutputTextFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CPackInputImage": "Image Input",
    "CPackInputString": "String Input",
    "CPackInputInt": "Int Input",
    "CPackInputFile": "File Input",
    "CPackInputAny": "Any Input",
    "CPackOutputImage": "Image Output",
    "CPackOutputAudio": "Audio Output",
    "CPackOutputVideo": "Video Output",
    "CPackOutputFile": "File Output",
    "CPackOutputZip": "Zip Output(img + txt file)",
    "CPackOutputZipSwitch": "Enable Zip Output",
    "CPackOutputTextFile": "Output Text to File",
}
