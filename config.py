from torchvision import transforms
import albumentations as A

class MyConfig:
    def __init__(self):
        # Model
        self.model_name = 'coatnet_rmlp_2_rw_384'
        
        # Hyperparameters
        self.train_batch_size = 16
        self.test_batch_size = 16
        self.lr = 0.00001
        self.epochs = 20
        self.num_classes = 500
        self.num_workers = 6

        # Transformations
        self.transform_type = "torchvision"
        self.image_size = 384
        self.train_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
            transforms.RandomRotation(15),  # 최대 15도 회전
            transforms.ColorJitter(brightness=0.2, contrast=0.2)  # 밝기 및 대비 조정
        ]

        # Optimizer
        self.optimizer_type = "Adam"
        self.patience = 4

        # Scheduler
        self.scheduler_type = 'StepLR'
        self.scheduler_gamma = 0.1 # StepLR, ReduceLROnPlateau에서 사용
        self.scheduler_step_multiplier = 2 # StepLR에서 사용
        self.scheduler_t_max = 10 # CosineAnnealingLR에서 사용

        # Ensemble
        self.model_configs = [{'name':'beitv2_large_patch16_224', 'input_size':(224, 224)},
                              {'name':'convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384', 'input_size':(384, 384)},
                              {'name':'deit_base_distilled_patch16_384', 'input_size':(384, 384)},
                              {'name':'eva02_large_patch14_448', 'input_size':(448, 448)},
                              #{'name':'deit3_small_patch16_384', 'input_size':(224, 224)},
                              {'name':'eva02_base_patch14_448', 'input_size':(448, 44)},
                              ]
    
    def get_config(self):
        config = {}
        for key, value in self.__dict__.items():
            if key == "train_transforms":
                config[key] = [instance.__class__.__name__ for instance in value]
            else:
                config[key] = value
        return config

my_config = MyConfig()