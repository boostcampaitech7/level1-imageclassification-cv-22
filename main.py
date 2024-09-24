from train import ModelTrainer
from inference import ModelInference
import torch
from config import my_config
import wandb
import random
import numpy as np

###################################
model_name = my_config.model_name
train_batch_size = my_config.train_batch_size
test_batch_size = my_config.test_batch_size

lr = my_config.lr
epochs = my_config.epochs
optimizer_type = my_config.optimizer_type
scheduler_type = my_config.scheduler_type
patience = my_config.patience
scheduler_gamma = my_config.scheduler_gamma # StepLR, ReduceLROnPlateau에서 사용
scheduler_step_multiplier = my_config.scheduler_step_multiplier # StepLR에서 사용
scheduler_t_max = my_config.scheduler_t_max # CosineAnnealingLR에서 사용
num_workers = my_config.num_workers

###################################

# Wandb 설정 !!!!!!
wandb.init(project="deit_coatnet_384", name=f"{model_name}")

# wandb에 변수 기록
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
    # Training process
    torch.cuda.empty_cache()
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
            num_workers=num_workers,
            patience=patience
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