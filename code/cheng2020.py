import os
import sys
import time
import random
import json

import torch
import torch.optim as optim
from compressai.layers import ResidualBlockWithStride, ResidualBlock, conv3x3, ResidualBlockUpsample, subpel_conv3x3
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from com_net.util.loss import RateDistortionLoss
from folder.img_folder import img_folder

from compressai.models.waseda import Cheng2020Anchor


class Cheng2020(Cheng2020Anchor):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(1, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 1, 2),
        )


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_optimizers(net, lr):
    parameters = {
        n
        for n, p in net.named_parameters()
        if p.requires_grad
    }
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    return optimizer


def train_one_epoch(
        model,
        criterion,
        train_dataloader,
        optimizer,
        epoch,
        clip_max_norm,
        writer):
    model.train()
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    cur_schedule = -.1

    for i, d in enumerate(train_dataloader):
        d = d.to(pms['device'])

        optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        loss.update(out_criterion["loss"])
        bpp_loss.update(out_criterion["bpp_loss"])
        mse_loss.update(out_criterion["mse_loss"])

        if round(100. * i / len(train_dataloader), 1) > cur_schedule:
            cur_schedule += .1
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.1f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.6f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.6f}'
            )
            writer.add_scalar('train_loss', loss.avg)


def test_one_epoch(
        model,
        criterion,
        test_dataloader,
        epoch,
        writer):
    model.eval()
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(pms['device'])

            out_net = model(d)
            out_criterion = criterion(out_net, d)

            loss.update(out_criterion["loss"])
            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.6f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tBpp loss: {bpp_loss.avg:.6f}"
    )
    writer.add_scalar('test_loss', loss.avg)

    return loss.avg


def main():
    train_dataset = img_folder(pms['root_path'], 'train')
    test_dataset = img_folder(pms['root_path'], 'test')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=pms['batch'],
        num_workers=pms['worker_num'],
        shuffle=True,
        pin_memory=(pms['device'] != "cpu")
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=pms['test_batch'],
        num_workers=pms['worker_num'],
        shuffle=False,
        pin_memory=(pms['device'] != "cpu")
    )

    net = Cheng2020(pms['N'])
    net = net.to(pms['device'])
    optimizer = create_optimizers(net, pms['learning_rate'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(pms['lmbda'])

    last_epoch = 0
    best_loss = float("inf")
    if pms['load_checkpoint']:
        print("Loading", pms['load_checkpoint'])
        try:
            checkpoint = torch.load(pms['load_checkpoint'], map_location=pms['device'])
            last_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["loss"]
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print("Loading finish")
        except FileNotFoundError:
            print("Loading failed")
    writer = SummaryWriter(pms['log'], purge_step=last_epoch)

    for epoch in range(last_epoch, pms['epochs']):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            pms['clip_max_norm'],
            writer
        )
        loss = test_one_epoch(
            net,
            criterion,
            test_dataloader,
            epoch,
            writer)
        lr_scheduler.step(loss)

        if loss < best_loss:
            best_loss = loss
            if pms['save']:
                if pms['save_checkpoint']:
                    filename = pms['save_checkpoint'] + pms['version'] + '.pth.tar'
                else:
                    filename = f'/home/kxfeng/iccv/checkpoint/baseline/cheng2020_{pms["lmbda"]}.pth.tar'
                print(f'Save model as {filename}')
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "pms": pms
                    },
                    filename)


pms = {
        'root_path': '/data/klin/vimeo_septuplet/sequences',
        'N': 128,
        'lmbda': 4096,
        'epochs': 10000,
        'learning_rate': 1e-4,
        'worker_num': 4,
        'batch': 16,
        'test_batch': 4,
        'device': 'cuda:6',
        'save': True,
        'seed': None,
        'clip_max_norm': 1.,
        'load_checkpoint': None,
        'save_checkpoint': None,
        'log': '/home/kxfeng/iccv/logs'
}

if __name__ == '__main__':
    print(f'current pid: {os.getpid()}')
    pms['load_checkpoint'] = f'/home/kxfeng/iccv/checkpoint/baseline/cheng2020_{pms["lmbda"]}.pth.tar'
    print(json.dumps(pms, indent=2))
    main()
