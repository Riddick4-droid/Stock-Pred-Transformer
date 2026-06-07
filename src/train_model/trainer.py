"""
Training Loops (Pre‑training & Fine‑tuning)
Provides Trainer classes that handle the full training cycle:
  - Data loading
  - Mixed‑precision (AMP)
  - Gradient clipping & NaN detection
  - Logging (console + optional MLflow)
  - Early stopping
  - Model checkpointing
Both PreTrainer and FineTuner read their settings from config.yaml.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional
import mlflow

from src.logger import logger
from src.exceptions import StockTransformerException
from src.utils import load_yaml, set_seed, get_device, EarlyStopping, check_nan_in_gradients, check_grad_norm,get_model_info
from src.compute_loss.losses import MultiTaskLoss
from src.models.stock_transformer import StockTransformer
from src.models.lora import inject_lora
from src.data.dataset import create_datasets

class TrainerBase:
    """Shared functionality for both pre‑training and fine‑tuning."""
    def __init__(self, config_path:str, mode:str):
        self.cfg = load_yaml(config_path)
        self.mode = mode
        self.train_cfg = self.cfg[self.mode]
        self.device = get_device()
        self.ft_cfg = self.cfg["fine_tuning"]
        self.ds_cfg = self.cfg["dataset"]
        set_seed(42)

        self.train_dataset, self.val_dataset, self.scaler, self.feature_cols = create_datasets(
            config_path=config_path, mode="pretrain" if mode == "pre_training" else "finetune"
        )
        batch_key = "batch_size_pre_train" if mode == "pre_training" else "batch_size_fine_tune"
        batch_size = self.cfg["dataset"].get(batch_key,4)
        num_workers = self.cfg["dataset"].get("num_workers",os.cpu_count())

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logger.info(f"{mode}: train={len(self.train_dataset)}, val={len(self.val_dataset)}")

        num_features = len(self.feature_cols)
        model_cfg = self.cfg["model"]
        self.model = StockTransformer(
            num_features=num_features,
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            d_ff=model_cfg["d_ff"],
            max_seq_len=model_cfg["max_seq_len"],
            dropout=model_cfg["dropout"]
        ).to(self.device)

        #loss and optimizer
        self.loss_fn = MultiTaskLoss(
            lambda_price=self.train_cfg.get("lambda_price", 1.0),
            lambda_dir=self.train_cfg.get("lambda_dir",0.5),
            lambda_vol=self.train_cfg.get("lambda_vol", 0.5),
            logvar_clamp_min=self.train_cfg.get("logvar_clamp_min", -10),
            logvar_clamp_max=self.train_cfg.get("logvar_clamp_max", 5)
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_cfg["lr"], 
                                           weight_decay=self.train_cfg.get("weight_decay",1e-4))
        self.scheduler = self._build_scheduler()

        self.early_stopping = EarlyStopping(
            patience=self.train_cfg.get("early_stopping_patience",3)
        )
        self.grad_clip_norm = self.train_cfg.get("grad_clip_norm",1.0)
        self.use_amp = self.train_cfg.get("use_amp", False)
        self.scaler_amp = torch.amp.GradScaler(self.device) if self.use_amp else None

        #logging
        self.checkpoint_dir = Path(self.cfg["data"]["checkpoints_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg["logging"].get("use_mlflow",False):
            mlflow.set_tracking_uri(self.cfg["logging"]["mlflow_tracking_uri"])
            mlflow.set_experiment(self.cfg["logging"]["experiment_name"])

        

    def _build_scheduler(self):
        if self.train_cfg.get("lr_scheduler", "cosine") == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.train_cfg["epochs"]
            )
        else:
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda e: 1.0)
        

    def save_checkpoint(self, val_loss: float, is_best: bool):
        ckpt = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "num_features": self.model.num_features,
            "config": self.cfg["model"]
        }
        path = self.checkpoint_dir / self.train_cfg["checkpoint_name"]
        torch.save(ckpt, path)
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(ckpt, best_path)
            logger.info(f"  -> New best model saved (val_loss={val_loss:.4f})")

    def train_epoch(self, dataloader, desc: str)->float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for batch in pbar:
            if self.mode == "pre_training":
                x, y_price = [b.to(self.device) for b in batch]
                dir_target = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                vol_target = torch.zeros(x.size(0), device=self.device)
            else:
                x,price_y, dir_y, vol_y = [b.to(self.device) for b in batch]
                y_price, dir_target, vol_target = price_y, dir_y, vol_y
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=str(self.device), enabled=self.use_amp):
                mu, logvar, dir_logits, vol_pred = self.model(x)
                loss, _ = self.loss_fn(mu, logvar, y_price, dir_logits, dir_target, vol_pred, vol_target)

            if self.scaler_amp:
                self.scaler_amp.scale(loss).backward()
            else:
                loss.backward()

            #NaN check
            if check_nan_in_gradients(self.model):
                logger.warning("NaN gradient detected; skipping batch update")
                self.optimizer.zero_grad()
                continue

            if self.scaler_amp:
                self.scaler_amp.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            if self.scaler_amp:
                self.scaler_amp.step(self.optimizer)
                self.scaler_amp.update()
            else:
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        return total_loss / len(dataloader)
    
    def validate_epoch(self, dataloader, desc: str) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, leave=False):
                if self.mode == "pre_training":
                    x, y_price = [b.to(self.device) for b in batch]
                    dir_target = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                    vol_target = torch.zeros(x.size(0), device=self.device)
                else:
                    x, price_y, dir_y, vol_y = [b.to(self.device) for b in batch]
                    y_price, dir_target, vol_target = price_y, dir_y, vol_y
                mu, logvar, dir_logits, vol_pred = self.model(x)
                loss, _ = self.loss_fn(mu, logvar, y_price,
                                       dir_logits, dir_target,
                                       vol_pred, vol_target)
                total_loss += loss.item()
        return total_loss / len(dataloader)
    

    def run(self,):
        self.current_epoch = 0
        best_val = float("inf")
        epochs = self.train_cfg["epochs"]

        if self.cfg["logging"].get("use_mlflow", False):
            mlflow.start_run(run_name=f"{self.mode}_{self.cfg['dataset'].get('finetune_tickers','all')}")
            # Log all relevant config sections as parameters
            mlflow.log_params({
                    # Model
                    "d_model": self.model_cfg["d_model"],
                    "n_heads":  self.model_cfg["n_heads"],
                    "n_layers":  self.model_cfg["n_layers"],
                    "d_ff":  self.model_cfg["d_ff"],
                    "dropout":  self.model_cfg["dropout"],
                    # Training
                    "lr": self.train_cfg["lr"],
                    "weight_decay": self.train_cfg.get("weight_decay", 0),
                    "epochs": self.train_cfg["epochs"],
                    "batch_size": self.ds_cfg.get("batch_size_pre_train",2),
                    "seq_len": self.cfg["dataset"]["seq_len"],
                    "grad_clip_norm": self.train_cfg.get("grad_clip_norm", 1.0),
                    # Loss weights
                    "lambda_price": self.train_cfg.get("lambda_price", 1.0),
                    "lambda_dir": self.train_cfg.get("lambda_dir", 0.5),
                    "lambda_vol": self.train_cfg.get("lambda_vol", 0.5),
                    # Mode specifics
                    "mode": self.mode,
                    "ticker": self.cfg["dataset"].get("finetune_tickers", "all") if self.mode == "fine_tuning" else "all"
                })
            if self.mode == "fine_tuning" and self.ft_cfg.get("use_lora", False):
                    mlflow.log_params({
                        "lora_r": self.ft_cfg["lora_r"],
                        "lora_alpha": self.ft_cfg["lora_alpha"],
                        "lora_dropout": self.ft_cfg["lora_dropout"],
                        "lr": self.ft_cfg["lr"],
                        "weight_decay": self.ft_cfg["weight_decay"],
                        "epochs": self.ft_cfg["epochs"],
                        "grad_clip_norm": self.ft_cfg["grad_clip_norm"],
                        "lr_scheduler": self.ft_cfg["lr_scheduler"],
                        "early_stopping_patience": self.ft_cfg["early_stopping_patience"],
                        "lambda_price": self.ft_cfg["lambda_price"],
                        "lambda_dir": self.ft_cfg["lambda_dir"],
                        "lambda_vol": self.ft_cfg["lambda_vol"],
                        "logvar_clamp_min": self.ft_cfg["logvar_clamp_min"],
                        "logvar_clamp_max": self.ft_cfg["logvar_clamp_max"],
                    })
        #train
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_loss = self.train_epoch(self.train_loader, f"Epoch {epoch}/{epochs} [Train]")
            val_loss = self.validate_epoch(self.val_loader, f"Epoch {epoch}/{epochs} [Val]")
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            if self.cfg["logging"].get("use_mlflow", False):
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
            self.save_checkpoint(val_loss, is_best)
            self.scheduler.step()

            if self.early_stopping(val_loss):
                logger.info("Early stopping triggered.")
                break

        if self.cfg["logging"].get("use_mlflow", False):
            mlflow.end_run()
        logger.info(f"Training finished. Best val loss: {best_val:.4f}")


class PreTrainer(TrainerBase):
    """Pre‑training on multiple stocks (only price head is active; others use dummy targets)."""
    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path, mode="pre_training")


class FineTuner(TrainerBase):
    """Fine‑tuning on a single stock (all heads + optional LoRA)."""
    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path, mode="fine_tuning")
        ft_cfg = self.cfg["fine_tuning"]

        # Load pre‑trained weights
        pretrain_path = self.checkpoint_dir / self.cfg["pre_training"]["checkpoint_name"]
        if not pretrain_path.exists():
            raise StockTransformerException(f"Pre‑trained checkpoint not found: {pretrain_path}")
        ckpt = torch.load(pretrain_path, map_location=self.device, weights_only=False)
        # Allow partial loading in case feature dimension changed (shouldn't, but safe)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Loaded pre‑trained weights from {pretrain_path}")

        # Inject LoRA if enabled
        if ft_cfg.get("use_lora", False):
            inject_lora(
                self.model,
                target_modules=ft_cfg.get("lora_target_modules", ["W_q","W_k","W_v","W_o"]),
                r=ft_cfg.get("lora_r", 8),
                alpha=ft_cfg.get("lora_alpha", 16.0),
                dropout=ft_cfg.get("lora_dropout", 0.05)
            )
            # Freeze everything except LoRA parameters and output heads
            for name, param in self.model.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
            for head_name in ['price_mu', 'price_logvar', 'direction_head', 'volatility_head']:
                for param in getattr(self.model, head_name).parameters():
                    param.requires_grad = True
            for param in self.model.final_ln.parameters():
                param.requires_grad = True
            # Recreate optimizer with only trainable parameters
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=ft_cfg["lr"],
                weight_decay=ft_cfg.get("weight_decay", 1e-4)
            )
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"LoRA trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


if __name__ == "__main__":
    # Quick self‑test: run a few steps of pre‑training (requires processed data)
    print("Trainer module loaded successfully.")