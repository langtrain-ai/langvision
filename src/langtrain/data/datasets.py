from torchvision import datasets, transforms

def get_dataset(name, data_dir, train, img_size=224):
    if name.lower() == 'cifar10':
        return datasets.CIFAR10(
            root=data_dir, train=train, download=True,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        )
    elif name.lower() == 'cifar100':
        return datasets.CIFAR100(
            root=data_dir, train=train, download=True,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        )
    else:
        raise ValueError(f"Unknown dataset: {name}") 