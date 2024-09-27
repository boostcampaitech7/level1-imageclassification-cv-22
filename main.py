import torch
import wandb
import random
import numpy as np

from config import my_config
from ensemble import ModelEnsemble
from inference import ModelInference
from train.train import ModelTrainer
# 학습 시 val_data를 사용하지 않은 ModelTrainer
#from train.without_val_train import ModelTrainer

model_name = my_config.model_name

# 하이퍼파라미터
train_batch_size = my_config.train_batch_size
test_batch_size = my_config.test_batch_size
lr = my_config.lr
epochs = my_config.epochs
num_classes = my_config.num_classes
num_workers = my_config.num_workers

# 옵티마이저
optimizer_type = my_config.optimizer_type
patience = my_config.patience

# 스케줄러
scheduler_type = my_config.scheduler_type
scheduler_gamma = my_config.scheduler_gamma # StepLR, ReduceLROnPlateau에서 사용
scheduler_step_multiplier = my_config.scheduler_step_multiplier # StepLR에서 사용
scheduler_t_max = my_config.scheduler_t_max # CosineAnnealingLR에서 사용

# Emsemble
model_configs = my_config.model_configs

# 학습여부 
train_pretrained = True
test_pretrained = False

# wandb
wandb.init(project="beit", name=f"{model_name}")
wandb.config.update(my_config.get_config())

def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Train
    train_model = True
    if train_model:
        print("Starting training process...")
        trainer = ModelTrainer(
            traindata_dir="/data/ephemeral/home/data/train", 
            traindata_info_file="/data/ephemeral/home/data/train.csv", 
            save_result_path="/data/ephemeral/home/results",
            model_name= model_name,
            batch_size= train_batch_size,
            lr= lr,
            pretrained=train_pretrained,
            epochs=epochs,
            optimizer_type=optimizer_type,               
            scheduler_gamma=scheduler_gamma,             
            scheduler_step_multiplier=scheduler_step_multiplier,
            scheduler_type=scheduler_type,
            scheduler_t_max=scheduler_t_max,
            num_workers=num_workers,
            patience=patience
        )
        trainer.run()
        print("Training completed.")

    # Inference
    run_inference = True
    
    if run_inference:
        print("Starting inference process...")
        inference_runner = ModelInference(
            testdata_dir="/data/ephemeral/home/data/test",
            testdata_info_file="/data/ephemeral/home/data/test.csv",
            save_result_path="/data/ephemeral/home/results",
            model_name= model_name,
            batch_size= test_batch_size,
            num_classes= num_classes,
            pretrained= test_pretrained,
            num_workers=num_workers
        )
        inference_runner.run_inference()
        print("Inference completed.")
    
    # Ensemble
    run_ensemble = False
    
    if run_ensemble:
        print("Starting inference process...")
        inference_runner = ModelEnsemble(
            testdata_dir="/data/ephemeral/home/data/test",
            testdata_info_file="/data/ephemeral/home/data/test.csv",
            save_result_path="/data/ephemeral/home/results",
            model_configs= model_configs,
            batch_size= test_batch_size,
            num_classes= num_classes,
            pretrained= test_pretrained,
            num_workers=num_workers
        )
        inference_runner.run_inference()
        print("Inference completed.")