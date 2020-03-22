from torch.autograd import Variable
import torch.utils.data
from PIL import Image
from torchvision import transforms as transforms
import numpy as np
from torch import nn

class_info = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 定义Net
class Net(nn.Module):
    '''
    AlexNet神经网络
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
    def __init__(self):
        self.model = None
        self.device = None
        self.test_transform = transforms.Compose([
            transforms.Resize((28, 28), 2),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def load_data(self, img_filename):
        imgo = Image.open(img_filename)
        img = self.test_transform(imgo).unsqueeze(0)
        img = Variable(img)
        return img

    def load_model(self):
        model_out_path = "model.pth"
        self.device = torch.device('cpu')
        self.model = torch.load(model_out_path, map_location='cpu')

    def run_single(self, img):
        img = self.load_data(img)
        output = self.model(img)
        prediction = torch.max(output, 1)
        class_ind = np.int32(prediction[1].cpu().numpy())
        res = class_info[class_ind[0]]
        return res

