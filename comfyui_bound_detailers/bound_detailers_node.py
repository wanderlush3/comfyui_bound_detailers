"""
BoundDetailer - A mask-targeted detailer node for ComfyUI.

This node is based on Impact Pack's FaceDetailer but adds a critical feature:
the ability to restrict detection and inpainting to only regions that overlap
with a user-supplied boundary mask.

Use case: In a multi-subject image, paint a mask over one subject and prompt
"green eyes" — only that subject's eyes will be modified. Without this node,
ALL detected eyes across all subjects would be changed.

Requires: ComfyUI-Impact-Pack (for core detection/enhancement infrastructure)

Mask Intersection Logic:
    The node uses PyTorch/NumPy tensor operations to compute the intersection
    between detected feature masks and the user-provided boundary mask.

    1. The bbox_detector runs on the FULL image and returns SEGS (segments)
       containing bounding boxes and per-feature masks.
    2. Optional SAM/segm refinement narrows the masks (same as FaceDetailer).
    3. CRITICAL STEP: core.segs_bitwise_and_mask(segs, boundary_mask) is called.
       This function:
       a) Converts the boundary mask to 2D via utils.make_2d_mask() — handles
          any batch/channel dimensions so it's (H, W).
       b) Scales both masks to uint8 [0-255].
       c) For each segment, crops the boundary mask to match the segment's
          crop_region coordinates: mask[y1:y2, x1:x2]
       d) Performs np.bitwise_and(segment_mask, cropped_boundary_mask).
          This zeros out any pixel in the segment mask that falls OUTSIDE
          the boundary mask.
       e) Normalizes back to float32 [0.0, 1.0] and creates a new SEG tuple.
    4. After intersection, any segments whose masks are entirely zero (i.e.,
       the detected feature was completely outside the boundary mask) are
       discarded.
    5. The surviving segments are passed to DetailerForEach.do_detail() for
       the actual KSampler inpainting, exactly as FaceDetailer does.
"""

import logging
import numpy as np
import torch

import nodes
from nodes import MAX_RESOLUTION

import comfy
import comfy.samplers

# Import Impact Pack core infrastructure
try:
    import impact.core as core
    from impact.core import SEG
    from impact.impact_pack import DetailerForEach
    import impact.utils as utils
except ImportError:
    raise ImportError(
        "[Bound Detailers] ComfyUI-Impact-Pack is required but not found.\n"
        "Please install it from: https://github.com/ltdrdata/ComfyUI-Impact-Pack"
    )


class BoundDetailer:
    """
    A mask-targeted detailer node that restricts feature detection/inpainting
    to only the region defined by a user-supplied boundary mask.

    This is functionally identical to FaceDetailer but with one critical
    addition: after detection, all segment masks are intersected with the
    boundary mask so that features outside the masked region are ignored.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL", {"tooltip": "The diffusion model for inpainting. Connect ImpactDummyInput to skip inference."}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (core.get_schedulers(),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),

                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),

                "sam_detection_hint": (
                    ["center-1", "horizontal-2", "vertical-2", "rect-4",
                     "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"],
                ),
                "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),

                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                "bbox_detector": ("BBOX_DETECTOR",),
                "mask": ("MASK", {"tooltip": "The boundary mask defining the permitted region for detailing. "
                                             "Only features overlapping this mask will be refined."}),
                "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),

                "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "sam_model_opt": ("SAM_MODEL",),
                "segm_detector_opt": ("SEGM_DETECTOR",),
                "detailer_hook": ("DETAILER_HOOK",),
                "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "scheduler_func_opt": ("SCHEDULER_FUNC",),
                "tiled_encode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "tiled_decode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "doit"

    CATEGORY = "BoundDetailers"

    DESCRIPTION = (
        "A mask-targeted detailer that restricts feature detection and inpainting "
        "to the region defined by the input mask. Based on Impact Pack's FaceDetailer "
        "but adds boundary mask intersection so you can control which subjects are "
        "affected (e.g., give one character green eyes and another blue eyes).\n\n"
        "The mask intersection is computed using bitwise AND between the detected "
        "feature masks and the user-supplied boundary mask. Features outside the "
        "boundary are ignored entirely."
    )

    @staticmethod
    def enhance_face(
        image, model, clip, vae, guide_size, guide_size_for_bbox, max_size,
        seed, steps, cfg, sampler_name, scheduler,
        positive, negative, denoise, feather, noise_mask, force_inpaint,
        bbox_threshold, bbox_dilation, bbox_crop_factor,
        sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
        sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size,
        bbox_detector, boundary_mask,
        segm_detector=None, sam_model_opt=None, wildcard_opt=None,
        detailer_hook=None,
        refiner_ratio=None, refiner_model=None, refiner_clip=None,
        refiner_positive=None, refiner_negative=None, cycle=1,
        inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None,
        tiled_encode=False, tiled_decode=False
    ):
        """
        Enhanced face/feature detailing with boundary mask intersection.

        This method mirrors FaceDetailer.enhance_face() but adds a critical step:
        after detection (and optional SAM/segm refinement), the detected segment
        masks are intersected with the user-supplied boundary_mask. Only features
        that overlap with the boundary mask proceed to the inpainting stage.

        Args:
            boundary_mask: A 2D or batched mask tensor defining the permitted
                          region. Features detected outside this mask are ignored.
            (all other args match FaceDetailer.enhance_face)

        Returns:
            Tuple of (enhanced_img, cropped_enhanced, cropped_enhanced_alpha,
                      mask, cnet_pil_list) — same as FaceDetailer.
        """

        # ================================================================
        # Step 1: Run feature detection on the full image
        # ================================================================
        bbox_detector.setAux('face')  # Default prompt for CLIPSeg compatibility
        segs = bbox_detector.detect(
            image, bbox_threshold, bbox_dilation, bbox_crop_factor,
            drop_size, detailer_hook=detailer_hook
        )
        bbox_detector.setAux(None)

        # ================================================================
        # Step 2: Optional SAM / segm mask refinement (same as FaceDetailer)
        # ================================================================
        if sam_model_opt is not None:
            sam_mask = core.make_sam_mask(
                sam_model_opt, segs, image,
                sam_detection_hint, sam_dilation,
                sam_threshold, sam_bbox_expansion,
                sam_mask_hint_threshold, sam_mask_hint_use_negative,
            )
            segs = core.segs_bitwise_and_mask(segs, sam_mask)

        elif segm_detector is not None:
            segm_segs = segm_detector.detect(
                image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size
            )

            if (hasattr(segm_detector, 'override_bbox_by_segm')
                    and segm_detector.override_bbox_by_segm
                    and not (detailer_hook is not None
                             and not hasattr(detailer_hook, 'override_bbox_by_segm'))):
                segs = segm_segs
            else:
                segm_mask = core.segs_to_combined_mask(segm_segs)
                segs = core.segs_bitwise_and_mask(segs, segm_mask)

        # ================================================================
        # Step 3: *** CRITICAL — Intersect with user boundary mask ***
        # ================================================================
        # This is the key difference from FaceDetailer.
        #
        # core.segs_bitwise_and_mask() performs per-segment intersection:
        #   - Converts boundary_mask to 2D (H, W) via utils.make_2d_mask()
        #   - Scales both masks to uint8 [0, 255]
        #   - For each segment, crops the boundary mask to the segment's
        #     crop_region: boundary[y1:y2, x1:x2]
        #   - Computes np.bitwise_and(segment_mask, cropped_boundary)
        #   - Normalizes result back to float32 [0.0, 1.0]
        #
        # After this, any segment whose detected feature was entirely
        # outside the boundary mask will have an all-zero mask.
        segs = core.segs_bitwise_and_mask(segs, boundary_mask)

        # ================================================================
        # Step 4: Filter out segments with empty masks
        # ================================================================
        # Segments that fell completely outside the boundary mask now have
        # all-zero cropped_masks. We discard them to avoid wasting compute
        # on empty inpainting operations.
        filtered_segs = []
        for seg in segs[1]:
            is_empty = (seg.cropped_mask == 0).all()
            if isinstance(is_empty, np.bool_):
                is_empty = bool(is_empty)
            elif isinstance(is_empty, torch.Tensor):
                is_empty = is_empty.item()

            if not is_empty:
                filtered_segs.append(seg)
            else:
                logging.info(
                    f"[Bound Detailers] Segment '{seg.label}' dropped "
                    f"(outside boundary mask)"
                )

        segs = (segs[0], filtered_segs)

        logging.info(
            f"[Bound Detailers] {len(filtered_segs)} segment(s) survived "
            f"boundary mask intersection"
        )

        # ================================================================
        # Step 5: Run the inpainting/enhancement pipeline on surviving segs
        # ================================================================
        if len(segs[1]) > 0:
            enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = \
                DetailerForEach.do_detail(
                    image, segs, model, clip, vae,
                    guide_size, guide_size_for_bbox, max_size,
                    seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, denoise, feather, noise_mask,
                    force_inpaint, wildcard_opt, detailer_hook,
                    refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                    refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                    refiner_negative=refiner_negative,
                    cycle=cycle, inpaint_model=inpaint_model,
                    noise_mask_feather=noise_mask_feather,
                    scheduler_func_opt=scheduler_func_opt,
                    tiled_encode=tiled_encode, tiled_decode=tiled_decode,
                )
        else:
            enhanced_img = image
            cropped_enhanced = []
            cropped_enhanced_alpha = []
            cnet_pil_list = []

        # Generate the combined output mask from all segments
        mask = core.segs_to_combined_mask(segs)

        if len(cropped_enhanced) == 0:
            cropped_enhanced = [utils.empty_pil_tensor()]

        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [utils.empty_pil_tensor()]

        if len(cnet_pil_list) == 0:
            cnet_pil_list = [utils.empty_pil_tensor()]

        return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list

    def doit(
        self, image, model, clip, vae, guide_size, guide_size_for,
        max_size, seed, steps, cfg, sampler_name, scheduler,
        positive, negative, denoise, feather, noise_mask, force_inpaint,
        bbox_threshold, bbox_dilation, bbox_crop_factor,
        sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion,
        sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size,
        bbox_detector, mask, wildcard, cycle=1,
        sam_model_opt=None, segm_detector_opt=None, detailer_hook=None,
        inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None,
        tiled_encode=False, tiled_decode=False
    ):
        """
        Main entry point called by ComfyUI.

        Processes each image in the batch independently (same behavior as
        FaceDetailer) and accumulates the results.
        """

        result_img = None
        result_mask = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []
        result_cnet_images = []

        if len(image) > 1:
            logging.warning(
                "[Bound Detailers] WARN: BoundDetailer is not designed for "
                "video detailing. For video, use Detailer For AnimateDiff."
            )

        for i, single_image in enumerate(image):
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, seg_mask, cnet_pil_list = \
                BoundDetailer.enhance_face(
                    single_image.unsqueeze(0), model, clip, vae,
                    guide_size, guide_size_for, max_size,
                    seed + i, steps, cfg, sampler_name, scheduler,
                    positive, negative, denoise, feather, noise_mask, force_inpaint,
                    bbox_threshold, bbox_dilation, bbox_crop_factor,
                    sam_detection_hint, sam_dilation, sam_threshold,
                    sam_bbox_expansion, sam_mask_hint_threshold,
                    sam_mask_hint_use_negative, drop_size,
                    bbox_detector, mask,
                    segm_detector_opt, sam_model_opt, wildcard, detailer_hook,
                    cycle=cycle, inpaint_model=inpaint_model,
                    noise_mask_feather=noise_mask_feather,
                    scheduler_func_opt=scheduler_func_opt,
                    tiled_encode=tiled_encode, tiled_decode=tiled_decode,
                )

            result_img = (
                torch.cat((result_img, enhanced_img), dim=0)
                if result_img is not None else enhanced_img
            )
            result_mask = (
                torch.cat((result_mask, seg_mask), dim=0)
                if result_mask is not None else seg_mask
            )
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)

        # Build the detailer_pipe output for downstream nodes
        pipe = (
            model, clip, vae, positive, negative,
            wildcard, bbox_detector, segm_detector_opt,
            sam_model_opt, detailer_hook, None, None, None, None
        )

        return (
            result_img,
            result_cropped_enhanced,
            result_cropped_enhanced_alpha,
            result_mask,
            pipe,
            result_cnet_images,
        )
