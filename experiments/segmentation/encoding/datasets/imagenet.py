

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

class ImageNetDataset(datasets.ImageFolder):
    BASE_DIR = "ILSVRC2012"
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), transform=None,
                 target_transform=None, train=True, **kwargs):
        split='train' if train == True else 'val'
        root = os.path.join(root, self.BASE_DIR, split)
        super(ImageNetDataset, self).__init__(
            root, transform, target_transform)
