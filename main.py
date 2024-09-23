import random
import numpy as np
import torch
from train import ModelTrainer
from inference import ModelInference
import wandb

###################################
model_name = 'deit_base_distilled_patch16_384'

train_batch_size = 32
test_batch_size = 32

lr = 0.001
epochs = 50

optimizer_type = 'Adam'
scheduler_type = 'ReduceLROnPlateau'
scheduler_gamma = 0.1 # StepLR, ReduceLROnPlateau에서 사용
scheduler_step_multiplier = 2 # StepLR에서 사용
scheduler_t_max = 10 # CosineAnnealingLR에서 사용
num_workers = 6

num_classes = 500
train_pretrained = True
test_pretrained = False
###################################

# Wandb 설정 !!!!!!
wandb.init(project="sketch_image_template_Ver", name=f"{model_name}")

# wandb에 변수 기록
config={
    "model_name": model_name,
    "train_batch_size": train_batch_size,
    "test_batch_size": test_batch_size,
    "lr": lr,
    "epochs": epochs,
    "optimizer_type": optimizer_type,
    "scheduler_type": scheduler_type,
    "scheduler_gamma": scheduler_gamma,
    "scheduler_step_multiplier": scheduler_step_multiplier,
    "scheduler_t_max": scheduler_t_max,
    "num_workers": num_workers,
    "num_classes": num_classes,
    "train_pretrained": train_pretrained,
    "test_pretrained": test_pretrained
}
wandb.config.update(config)

# 랜덤 시드 고정
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
    # Training process
    train_model = True  # Set this to False if you don't want to train
    
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
            num_workers=num_workers
        )
        # Run the training process
        trainer.run()
        print("Training completed.")

    # Inference process
    run_inference = True  # Set this to False if you don't want to run inference
    
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
        # Run the inference process
        inference_runner.run_inference()
        print("Inference completed.")