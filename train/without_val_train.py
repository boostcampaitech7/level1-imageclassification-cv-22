import torch
import pandas as pd
import torch.optim as optim

from config import my_config
from torch.utils.data import DataLoader
from dataset.dataset import CustomDataset

from loss.loss import Loss
from model.model_selector import ModelSelector

from trainer.without_val_cutmix_trainer import Trainer
from transform.transform_selector import TransformSelector

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
        train_info = pd.read_csv(self.traindata_info_file)

        self.num_classes = len(train_info['target'].unique())

        train_df = train_info

        transform_selector = TransformSelector(transform_type=my_config.transform_type)
        train_transform = transform_selector.get_transform(is_train=True)
        
        train_dataset = CustomDataset(
            root_dir=self.traindata_dir,
            info_df=train_df,
            transform=train_transform
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )
        self.val_loader = None


    def prepare_model(self):
        model_selector = ModelSelector(
            model_type='timm', 
            num_classes=self.num_classes,
            model_name=self.model_name, 
            pretrained=self.pretrained
        )
        self.model = model_selector.get_model()
        self.model.to(self.device)

    def set_optimizer_scheduler(self):
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
        elif self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.lr,
                weight_decay=5e-2  # 필요에 따라 weight_decay 조정 
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

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
                mode='min',
                factor=self.scheduler_gamma, 
                patience=self.patience
            )
        elif self.scheduler_type == 'CyclicLR':
            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.lr / 10,
                max_lr=self.lr,
                step_size_up=scheduler_step_size // 2,
                mode='triangular'
            )
        elif self.scheduler_type == 'ExponentialLR':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.scheduler_gamma
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

    def set_loss_function(self):
        self.loss_fn = Loss()

    def train(self):
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

        trainer.train()

    def run(self):
        self.prepare_data()
        self.prepare_model()
        self.set_optimizer_scheduler()
        self.set_loss_function()
        self.train()