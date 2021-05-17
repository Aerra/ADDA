from torchvision.datasets import MNIST, SVHN, USPS
import torchvision.transforms as transforms
import config
import os

def get_transform(kind):
    if kind == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif kind == "usps":
        transform = transforms.Compose(
            [
                #transforms.ToPILImage(),
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif kind == "svhn":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        raise ValueError(f"Unknown transform {kind}")
    
    return transform

class DatasetAccess:
    def __init__(self, kind, train=True):
        self.kind = kind
        self.train = train
        self.transform = get_transform(kind)

        self.datapath = str(config.DATA_DIR) + f"/datasets/{kind}"
        os.makedirs(self.datapath, exist_ok=True)


    def get_access(self):
        if self.kind == "mnist":
            return self.get_MNIST_dataset()
        elif self.kind == "usps":
            return self.get_USPS_dataset()
        elif self.kind == "svhn":
            return self.get_SVHN_dataset()

    def get_MNIST_dataset(self):
        MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]

        return MNIST(self.datapath, train=self.train, download=True, transform=self.transform)

    def get_USPS_dataset(self):
        return USPS(self.datapath, train=self.train, transform=self.transform, download=True)

    def get_SVHN_dataset(self):
        if self.train == True:
            train = "train"
        else:
            train = "test"
        return SVHN(self.datapath, split = train, transform=self.transform, download=True)