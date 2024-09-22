from train import ModelTrainer
from inference import ModelInference

###################################
model_name = 'resnet18'

train_batch_size = 64
test_batch_size = 64

lr = 0.001
epochs = 10

optimizer_type = 'Adam'
scheduler_type = 'StepLR'
scheduler_gamma = 0.1 # StepLR, ReduceLROnPlateau에서 사용
scheduler_step_multiplier = 2 # StepLR에서 사용
scheduler_t_max = 10 # CosineAnnealingLR에서 사용
num_workers = 4

num_classes = 500
train_pretrained = True
test_pretrained = False
###################################

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