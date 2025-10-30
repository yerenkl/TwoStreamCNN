import torchvision.transforms as T
from config import IMAGENET_MEAN, IMAGENET_STD

def spatial_train_tfms():
    return T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def spatial_eval_tfms():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def flow_tfms():
    return T.Compose([T.Resize(224)])
