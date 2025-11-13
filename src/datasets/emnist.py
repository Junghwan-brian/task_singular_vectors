import os

import torch

import torchvision
import torchvision.datasets as datasets


def rotate_img(img):
    return torchvision.transforms.functional.rotate(img, -90)


def flip_img(img):
    return torchvision.transforms.functional.hflip(img)


def emnist_preprocess():
    return torchvision.transforms.Compose(
        [
            rotate_img,
            flip_img,
        ]
    )


class EMNIST:
    def __init__(
        self,
        preprocess,
        location,
        batch_size=128,
        num_workers=8,
    ):
        # EMNIST-specific transforms must be applied FIRST (on PIL Image)
        # before any other transforms that might return a dict
        emnist_transforms = [rotate_img, flip_img]
        
        # Check if preprocess is a TwoAsymetricTransform or similar
        if hasattr(preprocess, 'weak_transform') and hasattr(preprocess, 'strong_transform'):
            # Apply EMNIST transforms inside both weak and strong transforms
            weak_transform = preprocess.weak_transform
            strong_transform = preprocess.strong_transform
            
            # Extract transforms from weak and strong, and prepend EMNIST transforms
            weak_transforms = weak_transform.transforms if hasattr(weak_transform, 'transforms') else [weak_transform]
            strong_transforms = strong_transform.transforms if hasattr(strong_transform, 'transforms') else [strong_transform]
            
            new_weak = torchvision.transforms.Compose(emnist_transforms + list(weak_transforms))
            new_strong = torchvision.transforms.Compose(emnist_transforms + list(strong_transforms))
            
            # Recreate the asymmetric transform with modified transforms
            class TwoAsymetricTransform:
                def __init__(self, weak, strong):
                    self.weak_transform = weak
                    self.strong_transform = strong
                def __call__(self, x):
                    return {"images": self.weak_transform(x), "images_": self.strong_transform(x)}
            
            preprocess = TwoAsymetricTransform(new_weak, new_strong)
        else:
            # Normal case: prepend EMNIST transforms
            if hasattr(preprocess, 'transforms'):
                # preprocess is a Compose, extract its transforms
                preprocess = torchvision.transforms.Compose(emnist_transforms + list(preprocess.transforms))
            else:
                # preprocess is a single transform
                preprocess = torchvision.transforms.Compose(emnist_transforms + [preprocess])
        # location = os.path.join(location, "EMNIST")
        self.train_dataset = datasets.EMNIST(
            root=location,
            download=False,
            split="digits",
            transform=preprocess,
            train=True,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.EMNIST(
            root=location,
            download=False,
            split="digits",
            transform=preprocess,
            train=False,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.train_dataset.classes
