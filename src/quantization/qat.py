import torch
import argparse
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from src.models.resnet20 import ResNet20, NoBNResNet20

def train(model, dataloader, optimizer, regularization=None, lmbda=0, device="cuda"):
    model.train()
    epoch_loss = 0
    count = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        preds = model(inputs)
        loss = torch.nn.functional.cross_entropy(preds, labels)

        if regularization != None:
            reg = 0
            if regularization == "l1":
                l1_norm = sum(param.abs().sum() for param in model.parameters())
                reg = lmbda * l1_norm
            if regularization == "l2":
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                reg = lmbda * l2_norm
            loss += reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        count += 1

    return epoch_loss / count

def eval(model, dataloader, device="cuda"):
    model.eval()
    epoch_loss = 0
    total = 0
    correct = 0
    count = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            preds = model(inputs)

            loss = torch.nn.functional.cross_entropy(preds, labels)
            epoch_loss += loss.item()

            preds = torch.argmax(preds, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            count += 1
    
    accuracy = 100 * correct / total
    return accuracy, epoch_loss / count 

def qat_eval(model, dataloader):

    device = "cpu"
    model = model.to(device)
    model.eval()

    print("pre conversion", flush=True)

    def print_size_of_model(model, label=""):
        torch.save(model.state_dict(), "temp.p")
        size=os.path.getsize("temp.p")
        print("model: ",label,' \t','Size (KB):', size/1e3, flush=True)
        os.remove('temp.p')
        return size
    
    print_size_of_model(model, label="")


    print("post conversion for eval", flush=True)
    q_model = torch.quantization.convert(model)
    print_size_of_model(q_model, label="")
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-reg", type=str)
    parser.add_argument("-lmbda", type=float)
    parser.add_argument("-save", type=str)
    parser.add_argument("-e", type=int, default=10)
    parser.add_argument("-m", type=str, default="mobile")
    parser.add_argument('--qat', action='store_true')
    args = parser.parse_args()
    print(args)

    device = "cuda"
    if args.m == "mobile":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=None)
    elif args.m == "linear":
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 10),
        )
    elif args.m == "linearbn":
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 10),
            torch.nn.BatchNorm1d(10)
        )
    elif args.m == "rn20":
        model = ResNet20(10, 3, dropout_rate=0.0)
    elif args.m == "nobnrn20":
        model = NoBNResNet20(10, 3, dropout_rate=0.0)

    if args.qat:
        model = torch.nn.Sequential(torch.quantization.QuantStub(), 
                  model, 
                  torch.quantization.DeQuantStub())
        
        backend = "fbgemm"
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
        model = torch.ao.quantization.prepare_qat(model)

    model = model.to(device)
    epochs = args.e
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.m == "mobile":
        transform = transforms.Compose([
            transform, 
            transforms.Resize(224)
        ])
    
    trainset = datasets.CIFAR10("../../data/CIFAR-10", train=True, transform=transform)
    testset = datasets.CIFAR10("../../data/CIFAR-10", train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 256, True)
    testloader = torch.utils.data.DataLoader(testset, 1024, False)

    for e in range(epochs):
        train_loss = train(model, trainloader, optimizer, args.reg, args.lmbda, device)
        test_acc, test_loss = eval(model, testloader, device)

        print(e, train_loss, test_loss, test_acc, flush=True)

    torch.save(model.state_dict(), args.save)

    if args.qat:
        qat_test_acc = qat_eval(model, testloader)
        print("QUANTIZED:", e, qat_test_acc, flush=True)

if __name__ == "__main__":
    main()