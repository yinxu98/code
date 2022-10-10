from torch.utils.data.dataloader import DataLoader

from .MyDatasetMECo import MyDatasetMECo
from .MyDatasetTest import MyDatasetTest


def build_data_loader_meco(cfg):
    dataset = MyDatasetMECo(cfg)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size.pretrain,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader

def build_data_loader_test(cfg, mode):
    dataset = MyDatasetTest(cfg, mode)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size[mode],
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader
