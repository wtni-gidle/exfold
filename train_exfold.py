from typing import Dict
import os
import logging
import argparse
import torch

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    LearningRateMonitor, 
    DeviceStatsMonitor, 
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies import DDPStrategy

from exfold.model.model2 import (
    ExFoldEnd2End, 
    ExFoldGeometry
)
from exfold.utils.loss import End2EndLoss, GeometryLoss
from exfold.data.data_modules import ExFoldDataModule
from exfold.utils.argparse_utils import remove_arguments
from exfold.utils.callbacks import EarlyStoppingVerbose, TQDMProgressBar4File
from exfold.utils.exponential_moving_average import ExponentialMovingAverage
from exfold.utils.lr_schedulers import AlphaFoldLRScheduler
from exfold.utils.metrics import lddt
from exfold.utils.feats import backbone_atom_fn
from exfold.utils.tensor_utils import tensor_tree_map
from exfold.config import model_config

logging.basicConfig(level=logging.INFO)


class ExFoldWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if self.config.globals.is_e2e:
            self.model = ExFoldEnd2End(config)
            self.loss = End2EndLoss(config.e2e_loss)

        elif self.config.globals.is_geom:
            self.model = ExFoldGeometry(config)
            self.loss = GeometryLoss(config.geom_loss)

        else:
            raise NotImplementedError
        
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )

        self.cached_weights = None
        self.last_lr_step = -1
    
    def forward(self, batch):
        return self.model(batch)
    
    def _log(
        self, 
        loss_breakdown: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor], 
        outputs: Dict[str, torch.Tensor], 
        train: bool = True
    ):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}_{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=not train, logger=True,
            )

            if train:
                self.log(
                    f"{phase}_{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                )
        
        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, 
                outputs,
            )
        for k, v in other_metrics.items():
            self.log(
                f"{phase}_{k}",
                torch.mean(v),
                on_step=False, on_epoch=True, logger=True
            )
    
    def _compute_validation_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metrics = {}
        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]

        C4_prime_gt, C4_prime_mask = backbone_atom_fn(
            atom_name="C4'",
            all_atom_positions=gt_coords,
            all_atom_mask=all_atom_mask,
        )
        C4_prime_pred = backbone_atom_fn(
            atom_name="C4'",
            all_atom_positions=pred_coords,
        )

        lddt_score = lddt(C4_prime_gt, C4_prime_pred, C4_prime_mask)
        metrics["lddt"] = lddt_score

        return metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_ids):
        if self.ema.device != self.device:
            self.ema.to(self.device)
        
        # Run the model
        outputs = self(batch)

        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=True)

        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_ids):
        # At the start of validation, load the EMA weights
        if self.cached_weights is None:
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])
        
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False)
    
    def validation_epoch_end(self, *args, **kwargs):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None
    
    def configure_optimizers(self) -> Dict:
        # Ignored as long as a DeepSpeed optimizer is configured
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            **self.config.train["optimizer"]
        )

        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            last_epoch=self.last_lr_step,
            **self.config.train["lr_scheduler"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }
    
    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)
    
    def on_load_checkpoint(self, checkpoint):
        self.ema.load_state_dict(checkpoint["ema"])
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
    
    def resume_last_lr_step(self, last_lr_step):
        self.last_lr_step = last_lr_step


def main(args):
    if args.seed is not None:
        seed_everything(args.seed, workers=True)

    # todo
    config = model_config()

    model_module = ExFoldWrapper(config)

    if args.resume_from_ckpt:
        sd = torch.load(args.resume_from_ckpt, map_location=torch.device('cpu'))
        if args.resume_model_weights_only:
            model_module.load_state_dict(sd)
            logging.info("Successfully loaded model weights...")
        else:
            last_lr_step = int(sd["lr_schedulers"][0]["last_epoch"]) - 1
            model_module.resume_last_lr_step(last_lr_step)
            logging.info("Successfully loaded last lr step...")
        
    # todo
    data_module = ExFoldDataModule(
        config=config.data,

    )
    data_module.prepare_data()
    data_module.setup()

    # region: callbacks
    callbacks = []

    model_summary = ModelSummary(max_depth=1)
    callbacks.append(model_summary)

    tqdm = TQDMProgressBar4File(refresh_rate=1)
    callbacks.append(tqdm)

    mc = ModelCheckpoint(
        # dirpath=None, # 默认{logger.log_dir}/checkpoints
        save_last=True, # saves an exact copy of the checkpoint to last.ckpt whenever a checkpoint file gets saved
        filename="Epoch_{epoch:03d}-TrL_{train_loss:.2f}-ValL_{val_loss:.2f}", #todo 待修改, 可能只要log有这些指标就行
        every_n_epochs=1, 
        save_top_k=-1, 
        auto_insert_metric_name=False,
        save_on_train_epoch_end=None, # 如果False, 在on_validation_end中check, 而不是在之后的on_train_epoch_end
    )
    callbacks.append(mc)

    # openfold的PerformanceLoggingCallback只能算每个batch的时间然后得到一个分布
    # simpleprofiler可以算出每一个重要的hook的耗时，但是必须要在fit结束之后才能将结果写入文件
    #! 既然是profiler，我们就可以尝试让trainer跑10个epoch, max_epoch=10，这样fit就能结束，profiler就有结果。
    #! 然后做好瓶颈分析后，就不需要在trainer中使用profiler了
    if args.profile:
        profiler = SimpleProfiler(
            dirpath=None,
            filename="performance_log",
            extended=True,
        )
        
        device_monitor = DeviceStatsMonitor()
        callbacks.append(device_monitor)
    else:
        profiler = None
    
    if args.early_stopping:
        es = EarlyStoppingVerbose(
            monitor="val_lddt",
            mode="max",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False, # 使用修改过的es，这里就要设置为False
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)
    
    if args.log_lr:
        # 会作用在lightningmodule.log()中
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    # endregion

    # region: loggers, 只考虑使用TensorBoardLogger或WandbLogger
    loggers = []
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "tb_logs"),
        name=args.exp_name, # "experiment"实际上是一组相关的运行。你可能会运行多个实验，每个实验都有不同的超参数设置或模型架构。
        version=args.exp_version, # 每个实验可能包含一个或多个运行，每个运行代表一次模型训练过程的记录。
        max_queue=10, # 指定了在内存中保存日志的数量，即内存队列的最大长度。超过这个值就会写入磁盘
        flush_secs=1200, # 超过刷新时间间隔就会写入磁盘。符合这两个条件之一就会将记录写入磁盘。值越大，内存需求越大，但io越少，训练速度越快
    )
    loggers.append(tb_logger)
    # endregion

    # region: strategy
    if (args.devices is not None and args.devices > 1) or args.num_nodes > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = None
    # endregion

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.output_dir, 
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        profiler=profiler,
    )

    if args.resume_model_weights_only:
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt
    
    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "train_ss_dir", type=str,
        help="Directory containing training secondary structure files"
    )
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_ss_dir", type=str, default=None,
        help="Directory containing precomputed validation ss"
    )
    
    parser = pl.Trainer.add_argparse_args(parser)

    ### Parameters that may be of interest
    # log_every_n_steps
    # check_val_every_n_epoch
    # accumulate_grad_batches
    # gradient_clip_val

    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )
    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(
        parser, 
        [
            "--accelerator", 
            "--gpus", 
            "--num_processes", 
            "--resume_from_checkpoint",
        ]
    )

    args = parser.parse_args()
    args.reload_dataloaders_every_n_epochs = 1
