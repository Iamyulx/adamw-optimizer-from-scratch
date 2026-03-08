import torch.nn as nn

class SmallMLP(nn.Module):
    
    def __init__ (self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.net(x)
