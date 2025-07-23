"""
Advanced Training Pipeline for Cancer Detection AI
Implements state-of-the-art training techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """
    Advanced training pipeline with modern techniques
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup optimizer with advanced techniques
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_loss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup AdamW optimizer with weight decay"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('scheduler_t0', 10),
            T_mult=2,
            eta_min=1e-7
        )
    
    def _setup_loss(self) -> nn.Module:
        """Setup focal loss for class imbalance"""
        return FocalLoss(
            alpha=self.config.get('focal_alpha', 1.0),
            gamma=self.config.get('focal_gamma', 2.0)
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.cuda() if torch.cuda.is_available() else images
            targets = targets.cuda() if torch.cuda.is_available() else targets
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': 100. * correct / total
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.cuda() if torch.cuda.is_available() else images
                targets = targets.cuda() if torch.cuda.is_available() else targets
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_acc': 100. * correct / total
        }
    
    def train(self, num_epochs: int):
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            logger.info(f"Epoch {epoch}: {metrics}")
            
            # Early stopping
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.get('patience', 10):
                logger.info("Early stopping triggered")
                break
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        import os
        os.makedirs('models/checkpoints', exist_ok=True)
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, f'models/checkpoints/{filename}')

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

def create_trainer(model: nn.Module, 
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  config: Dict) -> AdvancedTrainer:
    """Factory function to create trainer"""
    return AdvancedTrainer(model, train_loader, val_loader, config)

if __name__ == "__main__":
    print("Advanced Cancer AI Training Pipeline Ready!")
