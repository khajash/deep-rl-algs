from torch import nn, optim
from utils import get_conv2d_out_dim
from core import BaseNNAlg
from memory import ReplayMemory

class DQNPolicy(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        img_size: int,  
        n_actions: int,
    ) -> None:
        
        super().__init__()

        w = get_conv2d_out_dim(get_conv2d_out_dim(get_conv2d_out_dim(img_size, 8, 4), 4, 2), 3, 1)
        print(f"Image width after cnn = {w}")
        # TODO: make easy way to add batch norm to network
        self.policy = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(8,8), stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3*3*32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.policy(x)

