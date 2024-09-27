import albumentations as A

from torchvision import transforms

class MyConfig:
    
    """
    모델 학습, 테스트, 하이퍼파라미터 설정 등을 관리하는 설정 클래스입니다.
    
    이 클래스는 모델 이름, 배치 크기, 학습률, 에폭, 클래스 수 등의 하이퍼파라미터를 설정하고,
    이미지 변환, 옵티마이저, 스케줄러, 앙상블 모델 구성에 대한 설정을 포함합니다. 
    """
    
    def __init__(self):
        
        """
        MyConfig 클래스의 생성자 함수로, 각종 설정값을 초기화합니다.
        
        Args:
            None
        """
        
        self.model_name = 'eva02_base_patch14_448' 
        self.train_batch_size = 32
        self.test_batch_size = 32
        self.lr = 0.00001
        self.epochs = 1
        self.num_classes = 500
        self.num_workers = 6

        self.transform_type = "torchvision"
        self.image_size = 448
        use_grayscale = True
        self.train_transforms = [
            transforms.Grayscale(num_output_channels=3) if use_grayscale else transforms.Lambda(lambda x: x),
            transforms.AutoAugment(),
        ]

        self.optimizer_type = "Adam"
        self.patience = 4

        self.scheduler_type = 'ReduceLROnPlateau'
        self.scheduler_gamma = 0.1
        self.scheduler_step_multiplier = 2
        self.scheduler_t_max = 10

        self.model_configs = [
            {'name':'beitv2_large_patch16_224_origin', 'input_size':(224, 224)},
            {'name':'beitv2_large_patch16_224_cutmix', 'input_size':(224, 224)},
            {'name':'deit_base_distilled_patch16_384', 'input_size':(384, 384)},
            {'name':'deit_base_distilled_patch16_384_cutmix_train', 'input_size':(384, 384)},
            {'name':'eva02_base_patch14_448_cutmix', 'input_size':(448, 448)},
            {'name':'eva02_base_patch14_448_cutmix_train', 'input_size':(448, 448)},
            {'name':'eva02_base_patch14_448_origin', 'input_size':(448, 448)},
            {'name':'eva02_large_patch14_448_origin', 'input_size':(448, 448)},
        ]
    
    def get_config(self):
        
        """
        설정된 값을 딕셔너리 형태로 반환하는 함수입니다.
        
        Returns:
            dict: 설정값을 포함한 딕셔너리. 이미지 변환 설정은 클래스 이름으로 반환됩니다.
        """
        
        config = {}
        for key, value in self.__dict__.items():
            if key == "train_transforms":
                config[key] = [instance.__class__.__name__ for instance in value]
            else:
                config[key] = value
        return config

my_config = MyConfig()
