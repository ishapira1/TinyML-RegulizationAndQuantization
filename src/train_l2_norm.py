import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

from quantization import quantize_dynamic

class MLP(nn.Module):
    def __init__(self, dimensions):
        super().__init__()

        self.layerlist = nn.ModuleList()
        
        for idx in range(len(dimensions)-2):
            self.layerlist.append(nn.Linear(dimensions[idx], dimensions[idx+1]))
            self.layerlist.append(nn.ReLU())

        self.layerlist.append(nn.Linear(dimensions[-2], dimensions[-1]))

    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
        return x

def train(model, dataloader, optimizer, loss_fn, device="cuda"):
    model.train()   

    losses = 0
    total = 0
    for data,label in dataloader:
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)

        logits = model(data)
        loss = loss_fn(logits, label)

        loss.backward()
        optimizer.step()

        losses += loss.item()
        total += data.shape[0]
    return losses/total

def eval(model, dataloader, loss_fn, device="cuda"):
    model.eval()

    losses = 0
    total = 0
    for data, label in dataloader:
        data = data.to(device).to(torch.float32)
        label = label.to(device)
        with torch.no_grad():
            logits = model(data)
        loss = loss_fn(logits, label)
        losses += loss.item()
        total += data.shape[0]
    return losses/total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-epochs', type=int, default=10)
    args = parser.parse_args()
    print(args)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    train_dataset = FashionMNIST("datasets/", train=True, download=True, transform=transform)
    test_dataset = FashionMNIST("datasets/", train=False, download=True, transform=transform)
    input_shape = train_dataset[0][0].shape[0]

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=128)

    model = MLP([input_shape, 256, 256, 10]).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

    loss_fn = torch.nn.CrossEntropyLoss()

    # for epoch in range(args.epochs):
    #     train_loss = train(model, train_dataloader, optimizer, loss_fn, device=args.device)
    #     test_loss = eval(model, test_dataloader, loss_fn, device=args.device)

    #     print("Epoch: {} \t Train Loss: {} \t Test Loss: {}".format(epoch, train_loss, test_loss))
    #     torch.save(model.state_dict(), "models/l2_reg/epoch_{}_loss_{:.3f}.pt".format(epoch, test_loss))

    model = model.to("cpu")
    quantized_model = quantize_dynamic(model, layers_to_quantize=[nn.Linear, nn.ReLU])
    print("done")
    quantized_model(torch.randn(2,784))
    print("adsf")

    test_loss = eval(quantized_model, test_dataloader, loss_fn, device="cpu")

    print("Quantized Test Loss: {}".format(test_loss))
    torch.save(quantized_model.state_dict(), "models/l2_reg/quantized_loss_{:.3f}.pt".format(test_loss))


if __name__ == "__main__":
    main()