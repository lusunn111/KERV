"""Generate frozen-verifier supervision for the KERV drafter."""

import argparse

parser = argparse.ArgumentParser(description='Generate KERV drafter samples')

parser.add_argument('--base-model', required=True, help='OpenVLA checkpoint directory')
parser.add_argument('--data-root', required=True, help='RLDS dataset root')
parser.add_argument('--dataset-name', default='libero_goal_no_noops')
parser.add_argument('--output-dir', required=True)
parser.add_argument('--max-samples', type=int, default=0, help='0 means the full dataset')
args = parser.parse_args()
from pathlib import Path
from typing import Optional, Union

class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = args.base_model
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    use_spec: bool = False
gen_model_cfg=GenerateConfig()

class DataGenerationConfig:
    # fmt: off
    vla_path: str = args.base_model
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    image_aug: bool = True                                          # Whether to train with image augmentations
    # Directory Paths
    data_root_dir: Path = Path(args.data_root)
    dataset_name: str = args.dataset_name
    batch_size: int = 1                                          # Generation bsz
import os
import torch
from torch.utils.data import DataLoader
from openvla.prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from openvla.prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import AutoConfig, AutoImageProcessor
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from openvla.prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from openvla.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.openvla_utils import get_processor
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from accelerate import PartialState

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
distributed_state = PartialState()
torch.cuda.set_device(device_id := distributed_state.local_process_index)
torch.cuda.empty_cache()

cfg=DataGenerationConfig()
processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
action_tokenizer = ActionTokenizer(processor.tokenizer)
#Load大模型
quantization_config = None
print('loading vla')
model = get_model(gen_model_cfg)
processor = get_processor(gen_model_cfg)
print('loading data')
batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple([224, 224]),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
print('start enumerating')

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

#from transformers.modeling_outputs import CausalLMOutputWithPast
gen_model_cfg.unnorm_key = gen_model_cfg.task_suite_name
outdir = args.output_dir
sample_num = 0
write_sample_num = 0
for batch_idx, batch in enumerate(dataloader):
        action,token,hidden = model.predict_action(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    unnorm_key=gen_model_cfg.unnorm_key,
                    return_hidden_states=True,
                    do_sample=False
                )
        td={"input_ids":batch["input_ids"].cpu()[0],"pixel_values":batch["pixel_values"],"hidden_state":hidden,"loss_mask":batch["attention_mask"].cpu()[0],'predicted_tokens':token}
        hidden_state = torch.cat([item for item in td['hidden_state'][1]],dim=0).shape[0]
        origin_hidden_state = td['hidden_state'][0][0].shape[0]
        if hidden_state == origin_hidden_state + 6:
            writedata(outdir,td)
            write_sample_num += 1
        sample_num += 1
        if args.max_samples > 0 and sample_num >= args.max_samples:
            break
        if sample_num % 1000 == 0:
            print(sample_num)
        elif (sample_num==dataloader.__len__()):
            break
print('generation ended')
print('sample num',sample_num)
print('valid sample num',write_sample_num)
