import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim

from trainer.trainer import Trainer
from dataset.dataset import CustomDataset
from transform.transform_selector import TransformSelector
from model.model_selector import ModelSelector
from loss.loss import Loss

class ModelTrainer:
    def __init__(self, 
                 traindata_dir, 
                 traindata_info_file, 
                 save_result_path,
                 model_name='resnet18',
                 batch_size=64,
                 lr=0.001,
                 pretrained=True,
                 epochs=5,  
                 optimizer_type='Adam',
                 scheduler_gamma=0.1,
                 scheduler_step_multiplier=2,
                 scheduler_type='StepLR',
                 scheduler_t_max=10,
                 num_workers=6,
                 patience=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.traindata_dir = traindata_dir
        self.traindata_info_file = traindata_info_file
        self.save_result_path = save_result_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.pretrained = pretrained
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_step_multiplier = scheduler_step_multiplier
        self.scheduler_type = scheduler_type
        self.scheduler_t_max = scheduler_t_max
        self.num_workers = num_workers
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.loss_fn = None
        self.num_classes = 0
        self.patience = patience

        
    def prepare_data(self):
        # Load training data information from the CSV file.
        train_info = pd.read_csv(self.traindata_info_file)

        # Get the number of classes.
        self.num_classes = len(train_info['target'].unique())

        # Split the data into training and validation sets (80/20).
        train_df, val_df = train_test_split(
            train_info, 
            test_size=0.2,
            stratify=train_info['target']
        )

        # Set up transformations for training and validation.
        transform_selector = TransformSelector(transform_type="torchvision")
        train_transform = transform_selector.get_transform(is_train=True)
        val_transform = transform_selector.get_transform(is_train=False)

        # Create dataset instances for training and validation data.
        train_dataset = CustomDataset(
            root_dir=self.traindata_dir,
            info_df=train_df,
            transform=train_transform
        )
        val_dataset = CustomDataset(
            root_dir=self.traindata_dir,
            info_df=val_df,
            transform=val_transform
        )

        # Set up data loaders for batching and shuffling.
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )

    def prepare_model(self):
        # Initialize the model using the selected architecture and number of classes.
        model_selector = ModelSelector(
            model_type='timm', 
            num_classes=self.num_classes,
            model_name=self.model_name, 
            pretrained=self.pretrained
        )
        self.model = model_selector.get_model()

        # Move the model to the selected device (GPU/CPU).
        self.model.to(self.device)

    def set_optimizer_scheduler(self):
        # Initialize the optimizer based on the optimizer_type.
        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.lr
            )
        elif self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.lr, 
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # StepLR scheduler configuration.
        scheduler_step_size = len(self.train_loader) * self.scheduler_step_multiplier  # Decay every n epochs
        
        if self.scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=scheduler_step_size,
                gamma=self.scheduler_gamma
        )
        elif self.scheduler_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.scheduler_t_max
            )
        elif self.scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                factor=self.scheduler_gamma, 
                patience=10
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

    def set_loss_function(self):
        # Initialize the loss function.
        self.loss_fn = Loss()

    def train(self):
        # Initialize Trainer with model, optimizer, scheduler, etc.
        trainer = Trainer(
            model=self.model,
            device=self.device,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_fn=self.loss_fn,
            epochs=self.epochs,
            result_path=self.save_result_path,
            patience=self.patience
        )

        # Start training.
        trainer.train()

    def run(self):
        # Sequence of steps to run the full training process.
        self.prepare_data()
        self.prepare_model()
        self.set_optimizer_scheduler()
        self.set_loss_function()
        self.train()