import os
import json
import time

import torch
import torch.optim as optim
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from folder.spk_folder import spk_provider
from recon_net.spk2imgnet import new_Spike_Net_v3 as recon_net
from pms.train_recon_pms import pms


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

    def reset(self):
        self.__init__()


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
    t = AverageMeter()

    for i, d in enumerate(train_dataloader):
        s = time.time()
        isi = d[0].to(pms['device'])
        img = d[1].to(pms['device'])

        optimizer.zero_grad()
        out_net = model(isi)
        # recon = torch.clamp(out_net[0], 0, 1) ** (1 / 2.2)
        recon = torch.clamp(out_net[0], 0, 1)
        out_criterion = criterion(recon, img)
        out_criterion.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        loss.update(out_criterion)
        t.update(time.time() - s)

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion.item():.6f}'
                f'\tTime: {t.avg:.4f}'
            )
            t.reset()
            writer.add_scalar('train_loss', loss.avg)


def test_one_epoch(
        model,
        criterion,
        test_dataloader,
        epoch,
        writer):
    model.eval()
    loss = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            isi = d[0].to(pms['device'])
            img = d[1].to(pms['device'])
            out_net = model(isi)
            # recon = torch.clamp(out_net[0], 0, 1) ** (1 / 2.2)
            recon = torch.clamp(out_net[0], 0, 1)
            out_criterion = criterion(recon, img)

            loss.update(out_criterion)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.6f}"
    )
    writer.add_scalar('test_loss', loss.avg)

    return loss.avg


def main():
    train_dataset = spk_provider(pms['spk_path'], pms['gt_path'], 'train')
    test_dataset = spk_provider(pms['spk_path'], pms['gt_path'], 'test')
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

    net = recon_net(in_channels=13, features=64, out_channels=1, win_r=6, win_step=7)
    net = net.to(pms['device'])
    optimizer = create_optimizers(net, pms['learning_rate'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = nn.MSELoss()

    last_epoch = 0
    best_loss = float("inf")
    # print("Loading", pms['load_checkpoint'])
    # net.load_state_dict({k.replace('module.', ''): v for k, v in
    #                      torch.load('/home/kxfeng/tmp/model_061.pth', map_location=pms['device']).items()}, )
    if pms['load_checkpoint']:
        print("Loading", pms['load_checkpoint'])
        try:
            checkpoint = torch.load(pms['load_checkpoint'], map_location=pms['device'])
            last_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["loss"]
            net.load_state_dict(checkpoint["state_dict"])
            # net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items()}, )
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
                    filename = f'/home/kxfeng/iccv/checkpoint/recon_net.pth'
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


if __name__ == '__main__':
    print(f'current pid: {os.getpid()}')
    print(json.dumps(pms, indent=2))
    main()
