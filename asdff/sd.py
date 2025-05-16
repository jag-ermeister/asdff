from __future__ import annotations

from functools import cached_property

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInpaintPipeline,
    FluxPipeline,
    FluxFillPipeline,
)

from asdff.base import AdPipelineBase


class AdStableDiffusionPipeline(AdPipelineBase, StableDiffusionPipeline):
    # @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionPipeline


class AdCnPipeline(AdPipelineBase, StableDiffusionControlNetPipeline):
    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionControlNetInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionControlNetPipeline


class AdStableDiffusionXlPipeline(AdPipelineBase, StableDiffusionXLPipeline):

    def inpaint_pipeline(self):
        return StableDiffusionXLInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            scheduler=self.scheduler,
            image_encoder=self.image_encoder,
            feature_extractor=self.feature_extractor,
            force_zeros_for_empty_prompt=self.force_zeros_for_empty_prompt,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionXLPipeline


class AdFluxFillPipeline(AdPipelineBase, FluxPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_modules(**kwargs)

    def inpaint_pipeline(self):
        return FluxFillPipeline(
            transformer=self.transformer,
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
        )

    @property
    def txt2img_class(self):
        return FluxPipeline
