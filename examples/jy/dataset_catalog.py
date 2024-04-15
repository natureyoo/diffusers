"""Define datasets and their parameters."""
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# input:
#   root_path: ./data # CHANGE by yourself
#   dataset_name: ImageNetDataset
#   image_height: 64
#   image_width: 64
#   batch_size: 40 # how many data to fit in one gpu
#   disc_img_crop: 224
#   disc_img_resize: 232
#   sd_img_res: 512
#   use_objectnet: False
#   mean: [0.485, 0.456, 0.406]
#   std: [0.229, 0.224, 0.225]
#   subsample: 3 # number of samples per category
#   winoground_use_auth_token: null


class DatasetCatalog:
    def __init__(self, root_path='./data', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], disc_img_resize=232, disc_img_crop=224, sd_img_res=512, subsample=3, use_dit=False):
        ########### Define image transformations ###########
        interpolation = InterpolationMode.BILINEAR
        self.test_classification_transforms = T.Compose(
            [
                T.Resize(disc_img_resize, interpolation=interpolation),
                T.CenterCrop(disc_img_crop),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.test_diffusion_transforms = T.Compose(
            [
                T.Resize(disc_img_resize, interpolation=interpolation),
                T.CenterCrop(disc_img_crop),
                T.Resize(sd_img_res, interpolation=interpolation),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.classification_transforms = self.test_classification_transforms
        self.diffusion_transforms = self.test_diffusion_transforms

        ########### Define datasets ###########
        self.Food101Dataset = {   
            "target": "dataset_class_label.Food101Dataset",
            "train_params":dict(
                root=root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.Flowers102Dataset = {   
            "target": "dataset_class_label.Flowers102Dataset",
            "train_params":dict(
                root=root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.FGVCAircraftDataset = {   
            "target": "dataset_class_label.FGVCAircraftDataset",
            "train_params":dict(
                root=root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.OxfordIIITPetDataset = {   
            "target": "dataset_class_label.OxfordIIITPetDataset",
            "train_params":dict(
                root=root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.STL10Dataset = {   
            "target": "dataset_class_label.STL10Dataset",
            "train_params":dict(
                root=root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.CIFAR10Dataset = {   
            "target": "dataset_class_label.CIFAR10Dataset",
            "train_params":dict(
                root=root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.CIFAR100Dataset = {   
            "target": "dataset_class_label.CIFAR100Dataset",
            "train_params":dict(
                root=root_path,
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.ImageNetDataset = {   
            "target": "dataset_class_label.ImageNetDataset",
            "train_params":dict(
                root=root_path+'/ImageNet/val',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.ImageNetCDataset = {   
            "target": "dataset_class_label.ImageNetCDataset",
            "train_params":dict(
                root=root_path+'/ImageNet-C',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.ImageNetRDataset = {   
            "target": "dataset_class_label.ImageNetRDataset",
            "train_params":dict(
                root=root_path+'/imagenet-r',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.ImageNetStyleDataset = {   
            "target": "dataset_class_label.ImageNetStyleDataset",
            "train_params":dict(
                root=root_path+'/imagenet-styletransfer-v2/val',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.ImageNetADataset = {   
            "target": "dataset_class_label.ImageNetADataset",
            "train_params":dict(
                root=root_path+'/imagenet-a',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }
        
        self.ImageNetv2Dataset = {   
            "target": "dataset_class_label.ImageNetv2Dataset",
            "train_params":dict(
                root=root_path+'/imagenetv2-matched-frequency-format-val',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                subsample=subsample,
            ),
        }

        self.ObjectNetDataset = {
            "target": "dataset_class_label.ObjectNetDataset",
            "train_params":dict(
                root=root_path+'/ObjectNet/objectnet-1.0',
                classification_transform=self.classification_transforms,
                diffusion_transform=self.diffusion_transforms,
                test_classification_transform=self.test_classification_transforms,
                test_diffusion_transform=self.test_diffusion_transforms,
                use_dit=use_dit,
                subsample=subsample,
            ),
        }
