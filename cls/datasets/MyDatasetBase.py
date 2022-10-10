from PIL import Image
from torchvision import datasets, transforms


class MyDatasetBase(datasets.vision.VisionDataset):
    def __init__(self, cfg):
        transform = transforms.Compose([
            transforms.Resize([cfg.image_size, cfg.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.normalize.mean,
                std=cfg.normalize.std,
            ),
        ])

        super(MyDatasetBase, self).__init__(cfg.root, transform=transform)

    def _load_data(self, index):
        path, class_index = self.samples[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return img, class_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, gt = self._load_data(index % self.len)
        img = self.transform(img)
        return img, gt
