import os
import csv
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer 클래스는 모델 학습을 위한 클래스입니다. 

    Args:
        model (nn.Module): 학습할 모델.
        device (torch.device): 학습을 수행할 장치.
        train_loader (DataLoader): 학습 데이터 로더.
        val_loader (DataLoader): 검증 데이터 로더.
        optimizer (optim.Optimizer): 최적화 알고리즘.
        scheduler (optim.lr_scheduler): 학습률 스chedule.
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
        self.current_accuracy = 0
        self.patience = patience

    def save_model(self, epoch, loss):
        """
        모델을 저장하는 함수입니다.

        Args:
            epoch (int): 현재 에폭.
            loss (float): 현재 손실.
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

    def train_epoch(self):

        """
        한 에폭의 학습을 수행하는 함수입니다. 평균 손실을 반환하고, 제공된 스케줄러를 사용하여 학습률을 업데이트합니다.
        
        Returns:
            float: 평균 손실.
        """
        
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(self.train_loader)
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        else:
            self.scheduler.step()
        
        return avg_loss

    def validate(self) -> float:

        """
        검증 데이터를 사용하여 모델을 평가하는 함수입니다. 평균 손실을 계산하고, 최종 에폭에서 잘못된 예측을 저장합니다.

        Returns:
            float: 평균 손실.
        """

        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        incorrect_samples = []  
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                if self.current_epoch == self.epochs - 1:
                    incorrect_indices = (predicted != targets).nonzero(as_tuple=True)[0]
                    for idx in incorrect_indices:
                        incorrect_samples.append((targets[idx].cpu().item(), predicted[idx].cpu().item()))

                progress_bar.set_postfix(loss=loss.item())
                
        accuracy = correct_predictions / total_samples
        self.current_accuracy = accuracy
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        wandb.log({"val_loss": total_loss / len(self.val_loader), "val_accuracy": accuracy})
        
        if self.current_epoch == self.epochs - 1:
            csv_path = os.path.join(self.result_path, f'incorrect_samples_{self.model.model_name}.csv')
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['True Label', 'Predicted Label'])
                for true_label, predicted_label in incorrect_samples:
                    writer.writerow([true_label, predicted_label])
        
        return total_loss / len(self.val_loader)

    def train(self) -> None:
        
        """
        모델 학습을 수행하는 함수입니다. 각 에폭에서 학습 및 검증을 수행하고, 모델을 저장합니다.

        Returns:
            None
        """

        for epoch in range(self.epochs):
            self.current_epoch = epoch  
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")

            self.save_model(epoch, val_loss)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            wandb.log({"train_loss":train_loss, "val_loss":val_loss, "val_accuracy":self.current_accuracy})

        wandb.finish()