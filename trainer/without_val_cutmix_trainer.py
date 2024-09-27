import os
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    
    """
    검증 데이터를 사용하지 않고, CutMix를 사용하여 모델을 학습하는 클래스입니다.

    Args:
        model (nn.Module): 학습할 모델.
        device (torch.device): 학습을 수행할 장치.
        train_loader (DataLoader): 학습 데이터 로더.
        val_loader (DataLoader): 검증 데이터 로더.
        optimizer (optim.Optimizer): 최적화 알고리즘.
        scheduler (optim.lr_scheduler): 학습률 스케줄러.
        loss_fn (torch.nn.modules.loss._Loss): 손실 함수.
        epochs (int): 학습 에폭 수.
        result_path (str): 결과를 저장할 경로.
        patience (int): 조기 종료 카운터.
    """ 
    
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str,
        patience: int
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.best_models = []
        self.lowest_loss = float('inf')
        self.train_accuracy = 0
        self.current_accuracy = 0
        self.patience = patience
        self.scaler = GradScaler()
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')

    def save_model(self, epoch, loss):

        """
        모델을 저장하는 함수입니다.

        Args:
            epoch (int): 에폭 수.
            loss (float): 손실.
        """

        os.makedirs(self.result_path, exist_ok=True)

        model_path = os.path.join(self.result_path, self.model.model_name)
        os.makedirs(model_path, exist_ok=True)

        current_model_path = os.path.join(model_path, f'model_epoch_{epoch+1}_loss_{loss:.4f}_acc_{self.current_accuracy:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, self.model.model_name, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch+1}epoch result. Loss = {loss:.4f}")

    def rand_bbox(self, size, lam, scale_factor=0.5):

        """
        랜덤 박스를 생성하는 함수입니다.

        Args:
            size (tuple): 이미지 크기.
            lam (float): 랜덤 박스의 비율.
            scale_factor (float): 랜덤 박스의 크기 조정 인자.
        """

        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam) * scale_factor
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cutmix_data(self, x, y, alpha=1.0):

        """
        CutMix를 수행하는 함수입니다.

        Args:
            x (torch.Tensor): 입력 이미지.
            y (torch.Tensor): 레이블.
            alpha (float): CutMix의 강도.
        """

        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(x.size()[0])
        target_a = y
        target_b = y[rand_index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, target_a, target_b, lam

    def train_epoch(self):

        """
        학습 에폭을 수행하는 함수입니다.

        Args:
            None

        Returns:
            float: 평균 손실.
        """

        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            images, targets = batch[:2]
            images, targets = images.to(self.device), targets.to(self.device)

            images, targets_a, targets_b, lam = self.cutmix_data(images, targets, alpha=1.0)
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(images)
                loss = lam*self.loss_fn(outputs, targets_a) + (1-lam)*self.loss_fn(outputs, targets_b)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == targets).sum().item()
            total_samples += targets.size(0)
            
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_accuracy = correct_predictions / total_samples
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        else:
            self.scheduler.step()
        
        wandb.log({"train_loss": avg_loss, "train_accuracy": self.train_accuracy})
        print(f"Train Accuracy: {self.train_accuracy:.4f}")
        
        return avg_loss
    
    def train(self) -> None:

        """
        학습을 수행하는 함수입니다.

        Args:
            None

        Returns:
            None
        """
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss = self.train_epoch()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}\n")

            self.save_model(epoch, train_loss)
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step()

            wandb.log({"train_loss":train_loss, "train_accuracy":self.train_accuracy})
            
        wandb.finish()