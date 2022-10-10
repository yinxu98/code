import os

from cls.datasets.MyDatasetBase import MyDatasetBase


class MyDatasetTest(MyDatasetBase):
    def __init__(self, cfg, mode):
        super(MyDatasetTest, self).__init__(cfg)

        self._make_dataset(mode, cfg.percentage)

    def _make_dataset(self, mode, percentage):
        root = self.root

        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        sample_file = os.path.join(root, f'{mode}{percentage:d}.txt')
        with open(sample_file, 'r') as fin:
            ls_line = fin.readlines()
        ls_line = [line.strip('\n').split(',') for line in ls_line]
        samples = [(os.path.join(root, class_name,
                                 fname), class_to_idx[class_name])
                   for (class_name, fname) in ls_line]

        self.samples, self.class_to_idx = samples, class_to_idx
        self.len = len(samples)
