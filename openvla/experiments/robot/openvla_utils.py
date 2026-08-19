"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from contextlib import nullcontext

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from transformers.image_processing_utils import BatchFeature

from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig,SpecVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openvla.prismatic.extern.hf.modeling_speculation import SpecVLAforActionPrediction
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

#import the speculative decoding dependency
from openvla.specdecoding.model.cnets import MMModel

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoConfig.register("specvla", SpecVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

    # 使用transformers的from_pretrained加载模型
    print("[*] 使用本地OpenVLAForActionPrediction类并从预训练检查点加载")
    if cfg.use_spec:
        if cfg.spec_checkpoint is None:
            raise ValueError("cfg.spec_checkpoint must be provided when cfg.use_spec is True")
        print('load the vla model')
        #vla = OpenVLAforActionPrediction.from_pretrained(
        #    cfg.pretrained_checkpoint,
        #    load_in_8bit=cfg.load_in_8bit,
        #    load_in_4bit=cfg.load_in_4bit,
        #    low_cpu_mem_usage=True,
        #    trust_remote_code=True
            #use_spec = cfg.use_spec,
            #spec_checkpoint = cfg.spec_checkpoint
        #)
        vla = OpenVLAForActionPrediction.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_spec = True
        )
        #load_head_path=os.path.join(cfg.spec_checkpoint, "pytorch_model.bin")
        #print(load_model_path)
        #ea_layer_head = torch.load(load_head_path)
        initial_accept = getattr(cfg, "accept_threshold_start", cfg.accept_threshold)
        cfg.accept_threshold = initial_accept
        if cfg.parallel_draft:
            vla = SpecVLAforActionPrediction(
                base_model=vla,
                base_model_name_or_path=cfg.pretrained_checkpoint,
                ea_model_path=cfg.spec_checkpoint,
                parallel_draft=cfg.parallel_draft,
                accept_threshold=initial_accept,
            )
            #if cfg.parallel_draft:
            #    print('parallel drafter loaded')
        else:
            vla = SpecVLAforActionPrediction(
                base_model=vla,
                base_model_name_or_path=cfg.pretrained_checkpoint,
                ea_model_path=cfg.spec_checkpoint,
                accept_threshold=initial_accept,
            )

            #breakpoint()
        #head =
        #print('load the draft model')
        #load_model_path=os.path.join(cfg.spec_checkpoint, "pytorch_model.bin")
        #ea_layer_state_dict = torch.load(load_model_path)
        #print('reunify both models')
        #spec_vla = xx. xx  (vla,spec_head)
    else:
    # 使用from_pretrained直接加载模型
        vla = OpenVLAForActionPrediction.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    # 添加调试钩子
    # print("[*] 添加调试钩子")

    # 对predict_action方法添加调试钩子
    #original_predict_action = vla.predict_action

    #def debug_predict_action(*args, **kwargs):
        # print("\n=== 调用predict_action ===")
        # print(f"参数: {kwargs.keys()}")
        #if 'unnorm_key' in kwargs:
        #    pass
            # print(f"unnorm_key: {kwargs['unnorm_key']}")
        #result = original_predict_action(*args, **kwargs)
        # print(f"predict_action返回类型: {type(result)}")
        # if hasattr(result, 'shape'):
        #     print(f"predict_action返回shape: {result.shape}")
        # print("=== predict_action执行完毕 ===\n")
        #return result

    # 对generate方法添加调试钩子
    #original_generate = vla.generate

    #def debug_generate(*args, **kwargs):
        # print("\n=== 调用generate ===")
        # print(f"参数: {kwargs.keys()}")
        # print("max_new_tokens:", kwargs.get('max_new_tokens', 'not specified'))
    #    result = original_generate(*args, **kwargs)
        # print("调用self.generate方法")
        # print('1111111111111111')
        # print(f"generate返回类型: {type(result)}")
        # if hasattr(result, 'shape'):
        #     print(f"generate返回shape: {result.shape}")
        # print("=== generate执行完毕 ===\n")
    #    return result

    # 替换方法
    #vla.predict_action = debug_predict_action
    #vla.generate = debug_generate
    # print("已添加调试钩子到predict_action和generate方法")

    # Move model to device if not already
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    try:
        processor = AutoProcessor.from_pretrained(
            cfg.pretrained_checkpoint,
            trust_remote_code=True,
            local_files_only=True,
        )
    except OSError as exc:
        # In fully offline mode AutoProcessor may still try to resolve the dynamic module
        # from the Hugging Face Hub (see `auto_map` in `preprocessor_config.json`).
        # Falling back to the locally-registered implementation keeps the workflow offline.
        if "openvla/openvla-7b" in str(exc):
            preprocessor_config = os.path.join(cfg.pretrained_checkpoint, "preprocessor_config.json")
            if not os.path.isfile(preprocessor_config):
                raise

            with open(preprocessor_config, "r", encoding="utf-8") as f:
                processor_cfg = json.load(f)

            image_kwargs = {
                key: processor_cfg[key]
                for key in [
                    "use_fused_vision_backbone",
                    "image_resize_strategy",
                    "input_sizes",
                    "interpolations",
                    "means",
                    "stds",
                ]
                if key in processor_cfg
            }

            image_processor = PrismaticImageProcessor(**image_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.pretrained_checkpoint,
                trust_remote_code=True,
                local_files_only=True,
            )
            processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
        else:
            raise
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(
    vla,
    processor,
    base_vla_name,
    obs,
    task_label,
    unnorm_key,
    return_hidden_states=False,
    return_time=False,
    center_crop=False,
    generate_mode=None,
    accept_threshold=None,
    return_topk_index=False,
    token=None,
    *,
    history=None,
    step_idx=None,
    max_steps=None,
    dynamic_threshold=None,
    use_kalman=None,
    kalman_process_var=None,
    kalman_measurement_var=None,
    kalman_history_window=None,
    kalman_tree_enabled=None,
):
    """Generates an action with the VLA policy."""

    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    if center_crop:
        batch_size = 1
        crop_scale = 0.9
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, crop_scale, batch_size)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    if "openvla-v01" in base_vla_name:
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Tokenization is invariant within a task/episode while the image changes
    # every action.  Keep this optimization strictly on the input-preparation
    # side: the resulting input_ids and attention_mask are identical to the
    # original processor call, and unsupported processors use the old path.
    prompt_cache_enabled = bool(
        getattr(processor, "_flagos_prompt_token_cache_enabled", False)
    )
    pinned_image_enabled = bool(
        getattr(processor, "_flagos_pinned_image_buffer_enabled", False)
        and torch.cuda.is_available()
    )
    if (
        prompt_cache_enabled
        and processor.__class__.__name__ == "PrismaticProcessor"
        and hasattr(processor, "tokenizer")
        and hasattr(processor, "image_processor")
    ):
        token_cache = getattr(processor, "_flagos_prompt_token_cache", None)
        if token_cache is None:
            token_cache = {}
            processor._flagos_prompt_token_cache = token_cache
        cache_key = (prompt, str(getattr(processor, "_flagos_prompt_cache_model", "")))
        text_inputs = token_cache.get(cache_key)
        if text_inputs is None:
            text_inputs = processor.tokenizer(
                prompt, return_tensors="pt", padding=False, truncation=False
            )
            # Keep the cached tensors on CPU and detached.  A fresh BatchEncoding
            # is assembled below so the image tensor remains per-step data.
            text_inputs = BatchFeature(
                data={key: value.detach().clone() for key, value in text_inputs.items()}
            )
            token_cache[cache_key] = text_inputs
        image_inputs = processor.image_processor(image, return_tensors="pt")
        if pinned_image_enabled:
            pixel_values = image_inputs["pixel_values"]
            try:
                pinned_cache = getattr(processor, "_flagos_pinned_image_buffers", None)
                if pinned_cache is None:
                    pinned_cache = {}
                    processor._flagos_pinned_image_buffers = pinned_cache
                buffer_key = (tuple(pixel_values.shape), pixel_values.dtype)
                pinned = pinned_cache.get(buffer_key)
                if pinned is None:
                    pinned = torch.empty_like(pixel_values, pin_memory=True)
                    pinned_cache[buffer_key] = pinned
                pinned.copy_(pixel_values, non_blocking=False)
                image_inputs["pixel_values"] = pinned
                processor._flagos_pinned_image_buffer_hits = int(
                    getattr(processor, "_flagos_pinned_image_buffer_hits", 0)
                ) + 1
            except (RuntimeError, TypeError):
                # Pinned allocators are backend/platform dependent.  Keep the
                # original CPU tensor and ordinary H2D path if unavailable.
                processor._flagos_pinned_image_buffer_enabled = False
        # BatchEncoding is dict-compatible with the processor's output and
        # preserves the same keys and tensor values as a full processor call.
        inputs = BatchFeature(
            data={**text_inputs, **image_inputs}
        ).to(DEVICE, dtype=torch.bfloat16, non_blocking=True)
        processor._flagos_prompt_token_cache_hits = int(
            getattr(processor, "_flagos_prompt_token_cache_hits", 0)
        ) + 1
    else:
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Expose input-side counters to the model's per-action runtime manifest
    # without making the processor part of the model API.
    vla._flagos_prompt_token_cache_hits = int(
        getattr(processor, "_flagos_prompt_token_cache_hits", 0)
    )
    vla._flagos_pinned_image_buffer_hits = int(
        getattr(processor, "_flagos_pinned_image_buffer_hits", 0)
    )

    if return_hidden_states:
        action, token, hidden = vla.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            return_hidden_states=return_hidden_states,
            do_sample=False,
        )
        return action, token, hidden

    if return_topk_index:
        action, token, hidden = vla.eval_topk(
            **inputs,
            unnorm_key=unnorm_key,
            return_hidden_states=return_hidden_states,
            do_sample=False,
        )
        return action

    if hasattr(vla, "set_dynamic_threshold") and dynamic_threshold is not None:
        vla.set_dynamic_threshold(dynamic_threshold)
    if hasattr(vla, "set_step_context") and step_idx is not None:
        vla.set_step_context(step_idx, history)
    if hasattr(vla, "set_rollout_metadata"):
        vla.set_rollout_metadata(
            max_steps=max_steps,
            process_var=kalman_process_var,
            measurement_var=kalman_measurement_var,
            history_window=kalman_history_window,
            tree_enabled=kalman_tree_enabled,
        )
    if hasattr(vla, "set_kalman_enabled") and use_kalman is not None:
        vla.set_kalman_enabled(use_kalman)
    if hasattr(vla, "set_kalman_tree_enabled") and kalman_tree_enabled is not None:
        vla.set_kalman_tree_enabled(kalman_tree_enabled)

    # The model internally records CUDA events for phase timings.  The outer
    # action timer must not add a global pre-sync, otherwise work queued by
    # image transfer and the previous action is serialized before inference.
    cuda_start = None
    cuda_end = None
    start_time = time.perf_counter()
    if torch.cuda.is_available():
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record(torch.cuda.current_stream())
    inference_context = (
        torch.inference_mode()
        if bool(getattr(vla, "_flagos_inference_mode_enabled", False))
        else nullcontext()
    )
    with inference_context:
        action = vla.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            return_hidden_states=return_hidden_states,
            do_sample=False,
            generate_mode=generate_mode,
        )
    if cuda_end is not None:
        cuda_end.record(torch.cuda.current_stream())
        # The returned action is consumed by the CPU-side environment, so one
        # event wait is required.  It replaces the former two device-wide
        # synchronizations and does not block unrelated streams.
        cuda_end.synchronize()
        end_time = start_time + cuda_start.elapsed_time(cuda_end) / 1000.0
    else:
        end_time = time.perf_counter()
    if return_time:
        return action, (end_time, start_time)
    return action
