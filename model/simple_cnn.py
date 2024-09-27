import torch
import torch.nn as nn

class SimpleCNN(nn.Module):

    """
    간단한 CNN 아키텍처를 정의하는 클래스.

    Args:
        num_classes (int): 분류할 클래스의 개수. 
    """
    
    def __init__(self, num_classes: int):
    
        """
        SimpleCNN 클래스의 생성자.
        
        3개의 컨볼루션 레이어와 2개의 완전연결층을 포함한 CNN 아키텍처를 정의합니다.

        Args:
            num_classes (int): 분류할 클래스의 개수.
        """
    
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
     
        """
        순전파 함수. 입력 이미지를 CNN을 통해 전달하여 최종 출력값을 생성합니다.

        Args:
            x (torch.Tensor): 모델에 입력할 이미지 텐서. (배치 크기, 채널, 높이, 너비).

        Returns:
            torch.Tensor: 모델의 최종 출력값.
        """
    
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x