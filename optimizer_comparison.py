from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader

x, y = make_classification(
    n_samples=5000, 
    n_features=20, 
    n_classes=2
    )

x = torch.tensor(x).float()
y = torch.tensor(y).long()

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


def train(model, optimizer, epochs=10):

    loss_fn = nn.CrossEntropyLoss()
    
    losses = []

    for epoch in range(epochs):
        
        total = 0
        
        for x,y in loader:
            
            optimizer.zero_grad()
            
            pred = model(x)
            loss = loss_fn(pred,y)
            
            loss.backward()
            
            optimizer.step()
            
            total += loss.item()
        
        avg = total/len(loader)
        losses.append(avg)
        
        print(f"epoch {epoch} loss {avg}")
    
    return losses



import torch.optim as optim

results = {}

# SGD
model = SmallMLP()
opt = optim.SGD(model.parameters(), lr=0.01)
results["SGD"] = train(model,opt)

# Adam
model = SmallMLP()
opt = optim.Adam(model.parameters(), lr=1e-3)
results["Adam"] = train(model,opt)

# AdamW PyTorch
model = SmallMLP()
opt = optim.AdamW(model.parameters(), lr=1e-3)
results["AdamW_torch"] = train(model,opt)

# AdamW From Scratch
model = SmallMLP()
opt = AdamWFromScratch(model.parameters(), lr=1e-3)
results["AdamW_scratch"] = train(model,opt)




import matplotlib.pyplot as plt

for k, v in results.items():
    plt.plot(v, label=k)
    
plt.legend()
plt.title("Optimizer Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()    
