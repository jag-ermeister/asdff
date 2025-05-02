from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List, Mapping, Optional

from diffusers.utils import logging
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

from asdff.utils import (
    ADOutput,
    bbox_padding,
    composite,
    mask_dilate,
    mask_gaussian_blur,
)
from asdff.yolo import yolo_detector

logger = logging.get_logger("diffusers")


DetectorType = Callable[[Image.Image], Optional[List[Image.Image]]]


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


class AdPipelineBase(ABC):
    @property
    @abstractmethod
    def inpaint_pipeline(self) -> Callable:
        raise NotImplementedError

    @property
    @abstractmethod
    def txt2img_class(self) -> type:
        raise NotImplementedError

    def __call__(  # noqa: C901
        self,
        common: Mapping[str, Any] | None = None,
        txt2img_only: Mapping[str, Any] | None = None,
        inpaint_only: Mapping[str, Any] | None = None,
        images: Image.Image | Iterable[Image.Image] | None = None,
        detectors: DetectorType | Iterable[DetectorType] | None = None,
        mask_dilation: int = 4,
        mask_blur: int = 4,
        mask_padding: int = 32,
    ):
        if common is None:
            common = {}
        if txt2img_only is None:
            txt2img_only = {}
        if inpaint_only is None:
            inpaint_only = {}

        # This is not supported by Flux
        # if "strength" not in inpaint_only:
        #     inpaint_only = {**inpaint_only, "strength": 0.4}

        if detectors is None:
            detectors = [self.default_detector]
        elif not isinstance(detectors, Iterable):
            detectors = [detectors]

        if images is None:
            txt2img_output = self.process_txt2img(common, txt2img_only)
            txt2img_images = txt2img_output[0]
        else:
            if txt2img_only:
                msg = "Both `images` and `txt2img_only` are specified. if `images` is specified, `txt2img_only` is ignored."
                logger.warning(msg)

            txt2img_images = [images] if not isinstance(images, Iterable) else images

        init_images = []
        final_images = []

        for i, init_image in enumerate(txt2img_images):
            init_images.append(init_image.copy())
            final_image = None

            for j, detector in enumerate(detectors):
                masks = detector(init_image)
                if masks is None:
                    logger.info(
                        f"No object detected on {ordinal(i + 1)} image with {ordinal(j + 1)} detector."
                    )
                    continue

                for k, mask in enumerate(masks):
                    mask = mask.convert("L")
                    mask = mask_dilate(mask, mask_dilation)
                    # bbox = mask.getbbox()
                    # if bbox is None:
                    #     logger.info(f"No object in {ordinal(k + 1)} mask.")
                    #     continue
                    # mask = mask_gaussian_blur(mask, mask_blur)
                    # bbox_padded = bbox_padding(bbox, init_image.size, mask_padding)
                    #
                    # inpaint_output = self.process_inpainting(
                    #     common,
                    #     inpaint_only,
                    #     init_image,
                    #     mask,
                    #     bbox_padded,
                    # )

                    if not mask.getbbox():
                        logger.info(f"No object in {ordinal(k + 1)} mask.")
                        continue

                    mask = mask_gaussian_blur(mask, mask_blur)

                    inpaint_output = self.process_inpainting(
                        common,
                        inpaint_only,
                        init_image,
                        mask,
                        None,  # Unused now
                    )


                    inpaint_image = inpaint_output[0][0]

                    # final_image = composite(
                    #     init_image,
                    #     mask,
                    #     inpaint_image,
                    #     bbox_padded,
                    # )
                    final_image = inpaint_image
                    init_image = final_image

            if final_image is not None:
                final_images.append(final_image)

        return ADOutput(images=final_images, init_images=init_images)

    @property
    def default_detector(self) -> Callable[..., list[Image.Image] | None]:
        return yolo_detector

    def _get_txt2img_args(
        self, common: Mapping[str, Any], txt2img_only: Mapping[str, Any]
    ):
        return {**common, **txt2img_only, "output_type": "pil"}

    def _get_inpaint_args(
        self, common: Mapping[str, Any], inpaint_only: Mapping[str, Any]
    ):
        common = dict(common)
        pipe = self.inpaint_pipeline
        sig = inspect.signature(pipe)
        if (
            "control_image" in sig.parameters
            and "control_image" not in common
            and "image" in common
        ):
            common["control_image"] = common.pop("image")
        return {
            **common,
            **inpaint_only,
            "num_images_per_prompt": 1,
            "output_type": "pil",
        }

    def process_txt2img(
        self, common: Mapping[str, Any], txt2img_only: Mapping[str, Any]
    ):
        txt2img_args = self._get_txt2img_args(common, txt2img_only)
        return self.txt2img_class.__call__(self, **txt2img_args)

    # def process_inpainting(
    #     self,
    #     common: Mapping[str, Any],
    #     inpaint_only: Mapping[str, Any],
    #     init_image: Image.Image,
    #     mask: Image.Image,
    #     bbox_padded: tuple[int, int, int, int],
    # ):
    #     crop_image = init_image.crop(bbox_padded)
    #     crop_mask = mask.crop(bbox_padded)
    #     inpaint_args = self._get_inpaint_args(common, inpaint_only)
    #     inpaint_args["image"] = crop_image
    #     inpaint_args["mask_image"] = crop_mask
    #
    #     if "control_image" in inpaint_args:
    #         inpaint_args["control_image"] = inpaint_args["control_image"].resize(
    #             crop_image.size
    #         )
    #     pipe = self.inpaint_pipeline()
    #     return pipe(**inpaint_args)

    # Doing this for Flux Fill.  Is is totally necessary?
    # Use autocast to ensure image gets converted to float16 to match model weights
    # def process_inpainting(
    #         self,
    #         common: Mapping[str, Any],
    #         inpaint_only: Mapping[str, Any],
    #         init_image: Image.Image,
    #         mask: Image.Image,
    #         bbox_padded: tuple[int, int, int, int],
    # ):
    #     # Crop to the region of interest
    #     crop_image = init_image.crop(bbox_padded)
    #     crop_mask = mask.crop(bbox_padded)
    #
    #     print("🟡 bbox padded:", bbox_padded)
    #     print("🟡 init_image size:", init_image.size)
    #
    #     init_image.save("debug_init.png")
    #     crop_mask.save("debug_mask.png")
    #
    #     # Convert PIL to float16 tensors
    #     to_tensor = transforms.ToTensor()
    #     image_tensor = to_tensor(crop_image).unsqueeze(0).to(dtype=torch.float16, device="cuda")
    #     mask_tensor = to_tensor(crop_mask).unsqueeze(0).to(dtype=torch.float16, device="cuda")
    #
    #     inpaint_args = self._get_inpaint_args(common, inpaint_only)
    #     inpaint_args["image"] = image_tensor
    #     inpaint_args["mask_image"] = mask_tensor
    #
    #     # Resize and convert control_image if present
    #     if "control_image" in inpaint_args:
    #         control_img = inpaint_args["control_image"].resize(crop_image.size)
    #         control_tensor = to_tensor(control_img).unsqueeze(0).to(dtype=torch.float16, device="cuda")
    #         inpaint_args["control_image"] = control_tensor
    #
    #     # Sanity check
    #     print("🔍 image dtype:", inpaint_args["image"].dtype, inpaint_args["image"].device)
    #     print("🔍 mask dtype:", inpaint_args["mask_image"].dtype, inpaint_args["mask_image"].device)
    #
    #     # Call the inpainting pipeline
    #     pipe = self.inpaint_pipeline()
    #
    #     with torch.autocast("cuda", dtype=torch.float16):
    #         return pipe(**inpaint_args)

    # def process_inpainting(
    #         self,
    #         common: Mapping[str, Any],
    #         inpaint_only: Mapping[str, Any],
    #         init_image: Image.Image,
    #         mask: Image.Image,
    #         bbox_padded: tuple[int, int, int, int],  # <-- unused now
    # ):
    #     # 🚫 DO NOT CROP: FluxFill expects full-size inputs
    #     # Just convert the images and mask directly
    #
    #     # Convert PIL to float16 tensors on CUDA
    #     to_tensor = transforms.ToTensor()
    #     image_tensor = to_tensor(init_image).unsqueeze(0).to(dtype=torch.float16, device="cuda")
    #     mask_tensor = to_tensor(mask).unsqueeze(0).to(dtype=torch.float16, device="cuda")
    #
    #     # Merge args
    #     inpaint_args = self._get_inpaint_args(common, inpaint_only)
    #     inpaint_args["image"] = image_tensor
    #     inpaint_args["mask_image"] = mask_tensor
    #
    #     # Optional: Resize and convert control_image if present
    #     if "control_image" in inpaint_args:
    #         control_img = inpaint_args["control_image"].resize(init_image.size)
    #         control_tensor = to_tensor(control_img).unsqueeze(0).to(dtype=torch.float16, device="cuda")
    #         inpaint_args["control_image"] = control_tensor
    #
    #     # Sanity check
    #     print("🔍 image shape:", image_tensor.shape, image_tensor.dtype)
    #     print("🔍 mask shape:", mask_tensor.shape, mask_tensor.dtype)
    #
    #     # Run inpainting
    #     pipe = self.inpaint_pipeline()
    #     with torch.autocast("cuda", dtype=torch.float16):
    #         return pipe(**inpaint_args)

    def process_inpainting(
            self,
            common: Mapping[str, Any],
            inpaint_only: Mapping[str, Any],
            init_image: Image.Image,
            mask: Image.Image,
            bbox_padded: tuple[int, int, int, int],  # unused for Flux Fill
    ):
        # 🚫 DO NOT CROP: Flux Fill expects full-sized images and masks

        # ✅ Binarize the mask
        binary_mask = mask.point(lambda p: 255 if p > 128 else 0).convert("L")

        # 🧠 Convert to torch tensors
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(init_image).unsqueeze(0).to(dtype=torch.float16, device="cuda")
        mask_tensor = to_tensor(binary_mask).unsqueeze(0).to(dtype=torch.float16, device="cuda")

        # 📸 Debug outputs
        init_image.save("debug_full_input.png")
        binary_mask.save("debug_binary_mask.png")
        save_image(image_tensor, "debug_tensor_input.png")
        save_image(mask_tensor, "debug_tensor_mask.png")
        masked_image = image_tensor * (1 - mask_tensor)
        save_image(masked_image, "debug_tensor_masked_input.png")

        # 🛠 Prepare inpainting args
        inpaint_args = self._get_inpaint_args(common, inpaint_only)
        inpaint_args["image"] = image_tensor
        inpaint_args["mask_image"] = mask_tensor

        # Optional control image conversion
        if "control_image" in inpaint_args:
            control_img = inpaint_args["control_image"].resize(init_image.size)
            control_tensor = to_tensor(control_img).unsqueeze(0).to(dtype=torch.float16, device="cuda")
            inpaint_args["control_image"] = control_tensor

        # 🧪 Sanity checks
        print("🔍 image shape:", image_tensor.shape, image_tensor.dtype)
        print("🔍 mask shape:", mask_tensor.shape, mask_tensor.dtype)

        # 🌀 Run inpainting pipeline
        pipe = self.inpaint_pipeline()
        with torch.autocast("cuda", dtype=torch.float16):
            return pipe(**inpaint_args)
