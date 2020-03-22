import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import argparse
from torch import nn
from torchvision.datasets import mnist
import time
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="deep learning by PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs train')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=50, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()
    solver = Solver(args)
    solver.run()


# 定义Net
class Net(nn.Module):
    '''
    自定义AlexNet的神经网络
    '''
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.num_classes = None
        self.classe_names = None
        self.epoch = 0

    def load_data(self):

        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        root_path = './data'
        train_set = mnist.MNIST(root_path, train=True, transform=train_transform, download=True)
        test_set = mnist.MNIST(root_path, train=False, transform=test_transform, download=True)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)
        self.num_classes = len(train_set.classes)
        self.classe_names = train_set.classes
        print(train_set.classes)

    def load_model(self):
        if self.cuda:
            print('use gpu')
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            print('use cpu')
            self.device = torch.device('cpu')

        self.model = Net().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            batch_size_start = time.time()

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)
            total += target.size(0)

            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            print('%s Train, Epoch: %d/%d, BatchNum: %d/%d, Loss: %.4f | Acc: %.3f%%, (TrainCorrect:%d | Total:%d), CostTime :%.4f s'
                  % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                     self.epoch, self.epochs,
                     batch_num, len(self.train_loader), train_loss / (batch_num + 1),
                     100. * train_correct / total, train_correct, total, time.time() - batch_size_start))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                batch_size_start = time.time()

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                print('%s Test, Epoch: %d/%d, BatchNum: %d/%d, Loss: %.4f | Acc: %.3f%%, (TestCorrect:%d | Total:%d), CostTime :%.4f s'
                    % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                       self.epoch, self.epochs,
                       batch_num, len(self.test_loader),
                       test_loss / (batch_num + 1),
                       100. * test_correct / total, test_correct, total, time.time() - batch_size_start))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def save_best(self):
        model_out_path = "model_best.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            self.epoch = epoch
            print("\n===> epoch: %d/%d" % (epoch, self.epochs))
            train_result = self.train()
            test_result = self.test()
            if test_result[1] > accuracy:
                self.save_best()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


if __name__ == '__main__':
    main()
