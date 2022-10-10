import os

from cls.datasets.MyDatasetBase import MyDatasetBase


class MyDatasetMECo(MyDatasetBase):
    def __init__(self, cfg):
        super(MyDatasetMECo, self).__init__(cfg)

        self._make_dataset()

    def _make_dataset(self):
        root = self.root

        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        samples = []
        for class_name in classes:
            class_idx = class_to_idx[class_name]
            class_folder = os.path.join(root, class_name)
            if not os.path.isdir(class_folder):
                continue
            for folder, _, fnames in sorted(
                    os.walk(class_folder, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(folder, fname)
                    item = path, class_idx
                    samples.append(item)

        self.samples, self.class_to_idx = samples, class_to_idx
        self.len = len(samples)

    def __getitem__(self, index):
        img, _ = self._load_data(index % self.len)
        img = self.transform(img)
        return img
