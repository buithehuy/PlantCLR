import torchvision.transforms as transforms

class TwoCropTransform:
    """Create two crops of the same image for SimCLR."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

def get_simclr_transforms(img_size=384):
    """
    SimCLR strong stochastic augmentations.
    """
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    # Ensure kernel_size for GaussianBlur is an odd number
    kernel_size = int(0.1 * img_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return TwoCropTransform(data_transforms)

def get_downstream_transforms(img_size=384, mode='train'):
    """
    Standard augmentations for supervised fine-tuning.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
