from abc import ABC, abstractmethod
import argparse
import contextlib
import datetime
import functools
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Optional, Union
import warnings

from fairscale.nn.model_parallel import initialize as fs_init
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

try:
    from apex.optimizers import FusedAdam as AdamW
except ImportError:
    warnings.warn("cannot import FusedAdam from apex, use torch AdamW instead")
    from torch.optim import AdamW

from xllmx.data.dataset import FinetuneConversationDataset, ItemProcessorBase
from xllmx.data.sampler import FinetuneDistSampler
from xllmx.model.tokenizer import Tokenizer
import xllmx.util as util
import xllmx.util.lr_sched as lr_sched
import xllmx.util.misc as misc
from xllmx.util.tensor_type import promote_param_to_fp32


class FinetuneSolverBase(ABC):

    def __init__(self, args):
        self.args = args
        util.dist.init_distributed_mode(args)
        self.logger = self.configure_logger()
        self.logger.info(args)

        assert args.model_parallel_size == 1, (
            "Model parallelism currently not supported, ",
            "so please keep model_parallel_size to 1\n"
            "Note that model parallelism is different from and orthogonal to FSDP"
        )
        fs_init.initialize_model_parallel(args.model_parallel_size)
        self.global_rank = dist.get_rank()
        self.mp_rank = fs_init.get_model_parallel_rank()
        self.mp_world_size = fs_init.get_model_parallel_world_size()
        self.mp_group = fs_init.get_model_parallel_group()
        self.dp_rank = fs_init.get_data_parallel_rank()
        self.dp_world_size = fs_init.get_data_parallel_world_size()
        self.dp_group = fs_init.get_data_parallel_group()

        if self.args.auto_resume and self.args.resume_path is None:
            existing_checkpoints = [_ for _ in os.listdir(self.args.output_dir) if "epoch" in _]
            if len(existing_checkpoints) > 0:

                def ckpt_sort_key(s):
                    # divide ckpt directory names into epoch and iter parts
                    epoch, iteration = util.ckpt.split_ckpt_str_into_epoch_iter(s)
                    if iteration is None:
                        iteration = float("inf")
                    return epoch, iteration

                self.args.resume_path = os.path.join(
                    self.args.output_dir, sorted(existing_checkpoints, key=ckpt_sort_key)[-1]
                )
                self.logger.info(f"auto resume from {self.args.resume_path}")

        if args.output_dir and self.global_rank == 0:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        dist.barrier()

        if args.precision == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.logger.info("work dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        self.logger.info("{}".format(self.args).replace(", ", ",\n"))

        # define the model
        self.mixed_precision_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "tf32": torch.float32,
        }[self.args.precision]

        self.model, self.tokenizer, self.optimizer = self.build_model()

        self.dataset_train, self.sampler_train, self.dataloader_train = self.build_data()

        self.start_epoch = 0
        self.start_iter = 0
        self.metric_logger_to_resume = None

        if self.args.resume_path:
            self.resume(self.args.resume_path)

        if self.global_rank == 0:
            (Path(args.output_dir) / "tensorboard").mkdir(parents=True, exist_ok=True)
            self.log_writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "tensorboard"))
        else:
            self.log_writer = None

        gc.collect()
        torch.cuda.empty_cache()

    def configure_logger(self):
        rank = dist.get_rank()

        logger = logging.getLogger()

        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()  # Console handler
        f_handler = logging.FileHandler(Path(self.args.output_dir) / f"common.log")  # Rank-specific
        f_rank_handler = logging.FileHandler(
            Path(self.args.output_dir) / f"rank-{dist.get_rank()}.log"
        )  # Rank-specific

        # Console and common file handler captures all INFO and above messages
        c_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        f_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        f_rank_handler.setLevel(logging.INFO)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter(f"[rank{rank}:%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        f_rank_handler.setFormatter(formatter)
        # Set the log level based on the rank argument

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        logger.addHandler(f_rank_handler)

        return logger

    @classmethod
    def get_args_parser(cls):
        parser = argparse.ArgumentParser("xllmx Finetuning", add_help=False)

        # Schedule
        parser.add_argument(
            "--batch_size",
            default=4,
            type=int,
            help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
        )
        parser.add_argument(
            "--accum_iter",
            default=4,
            type=int,
            help="Accumulate gradient iterations " "(for increasing the effective batch size under memory constraints)",
        )
        parser.add_argument("--epochs", default=1, type=int)
        parser.add_argument("--warmup_epochs", type=float, default=0.03, help="epoch to warmup LR")

        # Optimizer parameters
        parser.add_argument("--lr", type=float, default=0.00002, help="learning rate (absolute lr)")
        parser.add_argument(
            "--min_lr", type=float, default=0.00, help="lower lr bound for cyclic schedulers that hit 0"
        )
        parser.add_argument("--wd", type=float, default=0.00, help="weight decay (default: 0.00)")
        parser.add_argument("--clip_grad", type=float, default=4.0, help="grad clipping norm")

        parser.add_argument("--init_from", default=None, type=str, help="path to checkpoint for model initialization")

        # Data parameters
        parser.add_argument("--data_config", default="/path/to/data/config/yaml", type=str, help="data config path")
        parser.add_argument(
            "--cache_ann_on_disk",
            action="store_true",
            help="cache the dataset annotations on disk to avoid duplication across ranks. "
            "can save CPU memory, especially with large datasets",
        )
        parser.add_argument(
            "--length_clustering",
            default=True,
            help="gather items with similar length to the same batch",
        )
        parser.add_argument("--disable_length_clustering", action="store_false", dest="length_clustering")
        parser.add_argument("--num_workers", default=8, type=int)
        parser.add_argument(
            "--pin_mem",
            action="store_true",
            help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
        )
        parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
        parser.set_defaults(pin_mem=True)

        # Seed
        parser.add_argument("--seed", default=0, type=int)

        # Control
        parser.add_argument("--output_dir", default="./output_dir", help="path for outputs")
        parser.add_argument("--save_interval", default=1, type=int, help="number of epochs between model saving")
        parser.add_argument(
            "--save_iteration_interval",
            default=5000,
            type=int,
            help="number of iterations between within-epoch model saving",
        )
        parser.add_argument(
            "--only_save_trainable", default=False, action="store_true", help="only save trainable model parameters"
        )
        parser.add_argument(
            "--ckpt_max_keep", default=2, type=int, help="maximum number of checkpoints to keep, <=0 means keep all"
        )
        parser.add_argument("--auto_resume", default=True, help="auto resume from args.output_dir")
        parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
        parser.add_argument("--resume_path", default=None, type=str, help="manually specify resume checkpoint")

        # Parallel
        parser.add_argument("--model_parallel_size", type=int, default=1)
        parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
        parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "tf32"], default="bf16")
        parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"], default="fp32")

        # Checkpointing
        parser.add_argument("--checkpointing", action="store_true", default=False, help="enable gradient checkpointing")
        # parser.add_argument('--quant', action="store_true", default=False,  # todo
        #                     help="enable quantization to speedup and save memory")

        return parser

    def build_model(self) -> (nn.Module, Tokenizer):
        init_from = self.args.resume_path or self.args.init_from
        if init_from is None:
            starting_point_path = Path(self.args.output_dir) / "starting_point"
            if dist.get_rank() == 0:
                if (starting_point_path / "config.json").exists():
                    self.logger.info(f"will use existing starting point at {starting_point_path}")
                    self.logger.info(
                        f"***********************************************************************\n"
                        f"********************************Caution********************************\n"
                        f"Caution: the starting point is created by some previous experiment run \n"
                        f"If the starting point saved by that run is broken, or if the expected  \n"
                        f"starting weights for the model has changed since that run, please manu-\n"
                        f"remove the saved path: \n"
                        f"{starting_point_path} \n"
                        f"and rerun the experiment.\n"
                        f"***********************************************************************\n"
                        f"***********************************************************************\n"
                    )
                else:
                    self.logger.info(f"creating starting-point weights at {starting_point_path}")
                    self._make_and_save_starting_point(save_path=str(starting_point_path))
            dist.barrier()
            init_from = str(starting_point_path)

        self.logger.info(f"Start instantiating unwrapped model from {init_from}")

        # only rank 0 instantiate, otherwise to meta
        unwrapped_model, tokenizer = self._model_func(init_from)

        if hasattr(unwrapped_model, "get_trainable_params"):
            trainable_params = dict(unwrapped_model.get_trainable_params())
            for key, param in unwrapped_model.named_parameters():
                if key in trainable_params:
                    param.requires_grad = True
                    promote_param_to_fp32(param)
                else:
                    param.requires_grad = False
                    keep_fp32_keywords = ["norm", "lm_head", "embed_tokens"]
                    if any([_ in key for _ in keep_fp32_keywords]):
                        promote_param_to_fp32(param)
                    elif param.is_floating_point():
                        param.data = param.data.to(self.mixed_precision_dtype)
        else:
            self.logger.warning(
                f"model class {type(unwrapped_model)} does not have `get_trainable_params` method,"
                f"set all params to trainable"
            )
            for key, param in unwrapped_model.named_parameters():
                param.requires_grad = True
                param.requires_grad = True
                promote_param_to_fp32(param)

        self.logger.info("Finish instantiating unwrapped model.")
        self.logger.info(f"Unwrapped model: \n{str(unwrapped_model)}")
        self.logger.info(f"Model config: \n{unwrapped_model.config.to_dict()}")

        # ----------------
        self.is_peft = getattr(unwrapped_model, "is_peft", False)  # todo
        self.logger.info(f"Model is Peft: {self.is_peft}")
        # ----------------

        misc.mark_mp_params(unwrapped_model)

        # defer this after FSDP
        misc.print_param_status(unwrapped_model)

        train_param_count_local, train_param_count_all = 0, 0
        frozen_param_count_local, frozen_param_count_all = 0, 0
        for name, param in unwrapped_model.named_parameters():
            model_parallel = getattr(param, "model_parallel", False)
            if param.requires_grad:
                if model_parallel:
                    train_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    train_param_count_all += param.numel()
                train_param_count_local += param.numel()
            else:
                if model_parallel:
                    frozen_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    frozen_param_count_all += param.numel()
                frozen_param_count_local += param.numel()

        self.logger.info(
            f"Trainable parameter count : {train_param_count_local} (local rank), {train_param_count_all} (all).\n"
            f"Frozen parameter count : {frozen_param_count_local} (local rank), {frozen_param_count_all} (all)."
        )

        # checkpointing (part1, should be called before FSDP wrapping)
        if self.args.checkpointing:
            # todo more hints for not-implemented
            checkpointing_list = unwrapped_model.get_checkpointing_wrap_module_list()
        else:
            checkpointing_list = []

        # todo pre-sync ignored states
        model = self.setup_fsdp_sync(
            unwrapped_model, self.args.data_parallel, self.args.precision, self.args.grad_precision
        )

        # broadcast non-model-parallel parameters within model parallel group
        misc.broadcast_nonmp_parameters(model)

        # checkpointing (part2, after FSDP wrapping)
        if self.args.checkpointing:
            print("apply gradient checkpointing")
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: submodule in checkpointing_list,
            )

        self.logger.info(f"Wrapped model: \n{str(model)}")

        # Setup optimizer
        opt = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.95))

        return model, tokenizer, opt

    @abstractmethod
    def _model_func(self, init_from: str) -> (nn.Module, Tokenizer | None):  # todo return type get finer # noqa
        raise NotImplementedError(f"{self.__class__} has to implement model_func for model instantiation")

    @abstractmethod
    def _make_and_save_starting_point(self, save_path: str):
        raise NotImplementedError(f"{self.__class__} has not implemented _make_and_save_starting_point()")

    def setup_fsdp_sync(self, model: nn.Module, data_parallel: str, precision: str, grad_precision: Optional[str]) -> FSDP:

        if self.dp_rank == 0:
            param_init_fn = None
        else:
            param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)


        model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
            ),
            process_group=fs_init.get_data_parallel_group(),
            sharding_strategy={
                "fsdp": ShardingStrategy.FULL_SHARD,
                "sdp": ShardingStrategy.SHARD_GRAD_OP,
            }[data_parallel],
            mixed_precision=MixedPrecision(
                param_dtype={
                    "fp32": torch.float,
                    "tf32": torch.float,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[precision],
                reduce_dtype={
                    "fp32": torch.float,
                    "tf32": torch.float,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[grad_precision or precision],
            ),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            limit_all_gathers=True,
            use_orig_params=True,
            param_init_fn=param_init_fn

        )
        torch.cuda.synchronize()

        return model

    def build_data(self):
        eff_batch_size = self.args.batch_size * self.args.accum_iter * fs_init.get_data_parallel_world_size()
        self.logger.info("effective batch size: %d" % eff_batch_size)
        dataset_train = self._dataset_func()
        self.logger.info(dataset_train)

        sampler_train = FinetuneDistSampler(
            dataset_train,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
            batch_size=self.args.batch_size,
            acc_grad=self.args.accum_iter,
            seed=self.args.seed,
            length_clustering=self.args.length_clustering,
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler_train,
            collate_fn=lambda batch: tuple(zip(*batch)),
            drop_last=True,
        )

        return dataset_train, sampler_train, dataloader_train

    @abstractmethod
    def _item_processor_func(self) -> ItemProcessorBase:
        raise NotImplementedError

    def _dataset_func(self):
        item_processor = self._item_processor_func()
        dataset = FinetuneConversationDataset(
            self.args.data_config, item_processor=item_processor, cache_on_disk=self.args.cache_ann_on_disk
        )
        return dataset

    def resume(self, resume_path: str):
        """
        Note: model ckpt is not loaded here because _model_func should already have met the resume path as init path
        """

        def _load_optimizer():
            opt_state_world_size = len(
                [x for x in os.listdir(resume_path) if x.startswith("optimizer.") and x.endswith(".pth")]
            )
            assert opt_state_world_size == dist.get_world_size(), (
                f"Resuming from a checkpoint with unmatched world size "
                f"({dist.get_world_size()} vs. {opt_state_world_size}) "
                f"is currently not supported."
            )
            self.logger.info(f"Resuming optimizer states from: {self.args.resume_path}")
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(
                        resume_path,
                        f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
                    ),
                    map_location="cpu",
                )
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.args.lr
                param_group["weight_decay"] = self.args.wd

        _load_optimizer()
        self.logger.info("Optimizer resume complete")

        resume_epoch, resume_iteration = util.ckpt.split_ckpt_str_into_epoch_iter(resume_path.split("/")[-1])

        if resume_iteration is None:
            self.start_epoch = resume_epoch + 1
            self.start_iter = 0
        else:
            self.start_epoch = resume_epoch
            self.start_iter = resume_iteration + 1

        self.logger.info(f"resume to epoch {self.start_epoch} iter {self.start_iter}")

        additional_rank_specific = os.path.join(
            resume_path, f"additional.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth"
        )
        if os.path.exists(additional_rank_specific):
            additional_rank_specific = torch.load(additional_rank_specific, map_location="cpu")
            if "metric_logger" in additional_rank_specific:
                self.metric_logger_to_resume = additional_rank_specific["metric_logger"]
                self.logger.info("metric logger resumed")

    def run(self):
        self.logger.info(f"Start training for {self.args.epochs} epochs")
        start_time = time.time()
        for epoch in range(self.start_epoch, self.args.epochs):
            self.dataloader_train.sampler.set_epoch(epoch, self.start_iter)  # todo rename set_epoch

            train_stats = self.train_one_epoch(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            if epoch % self.args.save_interval == 0 or epoch + 1 == self.args.epochs:
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    max_keep=self.args.ckpt_max_keep,
                )

            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            self.start_iter = 0
            self.metric_logger_to_resume = None

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Training time {}".format(total_time_str))

    def train_one_epoch(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        self.model.train(True)
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        accum_iter = self.args.accum_iter
        accum_counter = 0

        self.optimizer.zero_grad()
        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_train,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):
            accum_counter = (accum_counter + 1) % accum_iter
            is_gradient_accumulation_boundary = accum_counter == 0

            examples, labels = batch_data
            if is_gradient_accumulation_boundary or data_iter_step == start_iter:
                lr_sched.adjust_learning_rate_epoch(
                    self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
                )

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, additional_loss_dict = self.model(examples, labels)
            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            effective_loss = loss / accum_iter

            with (
                self.model.no_sync()
                if self.args.data_parallel in ["sdp", "hsdp"] and not is_gradient_accumulation_boundary
                else contextlib.nullcontext()
            ):
                effective_loss.backward()

            if is_gradient_accumulation_boundary:
                grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                metric_logger.update(grad_norm=grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            for metric_name, metric in metric_logger.meters.items():
                metric_value = metric.value
                metric_value = util.dist.all_reduce_mean(metric_value)
                if log_writer is not None:
                    log_writer.add_scalar(
                        metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                    )

            # save within epoch
            n_update_per_save = self.args.save_iteration_interval // accum_iter
            if (
                is_gradient_accumulation_boundary and ((data_iter_step + 1) // accum_iter) % n_update_per_save == 0
            ) or (data_iter_step + 1 == accum_iter and epoch == 0):
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    iteration=data_iter_step,
                    additional_rank_specific={
                        "metric_logger": metric_logger,
                    },
                    max_keep=self.args.ckpt_max_keep,
                )

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
