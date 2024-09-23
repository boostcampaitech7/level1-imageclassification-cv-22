import os

import torch
import torch.nn as nn
import torch.optim as optim
import csv
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

import wandb

class Trainer:
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
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.result_path = result_path  # 모델 저장 경로
        self.best_models = [] # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.lowest_loss = float('inf') # 가장 낮은 Loss를 저장할 변수
        self.current_accuracy = 0
        self.patience = patience

    def save_model(self, epoch, loss):
        # 모델 저장 경로 설정
        os.makedirs(self.result_path, exist_ok=True)

        # 현재 에폭 모델 저장
        model_path = os.path.join(self.result_path, self.model.model_name)
        os.makedirs(model_path, exist_ok=True)
        current_model_path = os.path.join(model_path, f'model_epoch_{epoch+1}_loss_{loss:.4f}_acc_{self.current_accuracy:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        # 최상위 3개 모델 관리
        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # 가장 높은 손실 모델 삭제
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        # 가장 낮은 손실의 모델 저장
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, self.model.model_name, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch+1}epoch result. Loss = {loss:.4f}")

    def train_epoch(self):
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
        
        # ReduceLROnPlateau 스케줄러의 경우, 메트릭을 전달
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        else:
            self.scheduler.step()
        
        return avg_loss

    def validate(self) -> float:
        # 모델의 검증을 진행
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        incorrect_samples = []  # 틀린 샘플을 저장할 리스트
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)    
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                
                # 정확도 계산
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                # 마지막 에포크에서 틀린 샘플 저장
                if self.current_epoch == self.epochs - 1:
                    incorrect_indices = (predicted != targets).nonzero(as_tuple=True)[0]
                    for idx in incorrect_indices:
                        incorrect_samples.append((targets[idx].cpu().item(), predicted[idx].cpu().item()))

                progress_bar.set_postfix(loss=loss.item())
                
        
        accuracy = correct_predictions / total_samples
        self.current_accuracy = accuracy
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # wandb 로그 기록
        wandb.log({"val_loss": total_loss / len(self.val_loader), "val_accuracy": accuracy})
        
        # 마지막 에포크에서 틀린 샘플을 CSV로 저장
        if self.current_epoch == self.epochs - 1:  # 마지막 에포크인지 확인
            csv_path = os.path.join(self.result_path, f'incorrect_samples_{self.model.model_name}.csv')
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['True Label', 'Predicted Label'])  # 헤더 수정
                for true_label, predicted_label in incorrect_samples:
                    writer.writerow([true_label, predicted_label])  # 이미지 대신 라벨값 저장
        
        return total_loss / len(self.val_loader)

    def train(self) -> None:
        # 전체 훈련 과정을 관리
        for epoch in range(self.epochs):
            self.current_epoch = epoch  # 현재 에포크 저장
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")

            self.save_model(epoch, val_loss)
            self.scheduler.step(val_loss)

            wandb.log({"train_loss":train_loss, "val_loss":val_loss, "val_accuracy":self.current_accuracy})

        wandb.finish()
