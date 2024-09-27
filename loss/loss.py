import torch
import torch.nn as nn

class Loss(nn.Module):

    """
    모델의 손실함수를 계산하는 클래스.

    Attributes:
        loss_fn (nn.CrossEntropyLoss): CrossEntropyLoss 손실 함수.
    """

    def __init__(self):

        """
        Loss 클래스의 생성자.
        nn.CrossEntropyLoss를 초기화합니다.
        """
        
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:

        """
        모델의 예측값과 실제 레이블을 입력받아 손실값을 계산합니다.

        Args:
            outputs (torch.Tensor): 모델의 출력값 (예측값).
            targets (torch.Tensor): 실제 레이블값.

        Returns:
            torch.Tensor: 계산된 손실값.
        """
        
        return self.loss_fn(outputs, targets)
    