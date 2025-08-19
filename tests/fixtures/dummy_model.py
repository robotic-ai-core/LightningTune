"""
Dummy PyTorch Lightning model for testing.
"""

import torch
import torch.nn as nn
try:
    import lightning.pytorch as pl
except ImportError:
    import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DummyModel(L.LightningModule):
    """Simple model for testing optimization."""
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 32,
        output_dim: int = 2,
        learning_rate: float = 0.001,
        dropout: float = 0.1,
        optimizer_type: str = "adam",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # Initialize weights to prevent exploding gradients
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization with small scale."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization to prevent exploding gradients
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        self.training_step_outputs.append({"loss": loss.detach(), "acc": acc})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        self.validation_step_outputs.append({"val_loss": loss.detach(), "val_acc": acc})
        return {"val_loss": loss, "val_acc": acc}
    
    def on_train_epoch_end(self):
        if self.training_step_outputs:
            avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
            avg_acc = torch.stack([x["acc"] for x in self.training_step_outputs]).mean()
            self.log("train_loss_epoch", avg_loss)
            self.log("train_acc_epoch", avg_acc)
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
            avg_acc = torch.stack([x["val_acc"] for x in self.validation_step_outputs]).mean()
            self.log("val_loss_epoch", avg_loss)
            self.log("val_acc_epoch", avg_acc)
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        if self.hparams.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        elif self.hparams.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def on_before_optimizer_step(self, optimizer):
        """Clip gradients to prevent exploding gradients."""
        # Clip gradients to prevent nan/inf
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)


class DummyDataModule(L.LightningDataModule):
    """Simple data module for testing."""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_samples: int = 1000,
        input_dim: int = 10,
        num_classes: int = 2,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.val_split = val_split
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage=None):
        # Generate random data
        np.random.seed(42)
        X = np.random.randn(self.num_samples, self.input_dim).astype(np.float32)
        y = np.random.randint(0, self.num_classes, self.num_samples)
        
        # Split into train/val
        split_idx = int(self.num_samples * (1 - self.val_split))
        
        X_train = torch.from_numpy(X[:split_idx])
        y_train = torch.from_numpy(y[:split_idx])
        
        X_val = torch.from_numpy(X[split_idx:])
        y_val = torch.from_numpy(y[split_idx:])
        
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )