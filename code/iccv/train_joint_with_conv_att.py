import os
import json
import shutil
import sys
import time
import argparse

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from whole_net.joint_net import Joint_Net_with_lstm_att, Joint_Net_with_conv_att
from com_net.util.loss import RateDistortionLoss, RDDLoss

from pms.joint.recon_cheng128 import pms
from folder.spk_folder import spk_provider


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
        clip_max_norm):
    model.train()
    t = AverageMeter()
    cur_schedule = 0
    cur_save_schedule = 9.9

    for i, d in enumerate(train_dataloader):
        s = time.time()
        isi = d[0].to(pms['device'])
        img = d[1].to(pms['device'])

        optimizer.zero_grad()
        out_net = model(isi)
        # recon = torch.clamp(out_net, 0, 1)
        out_criterion = criterion(out_net, img)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        t.update(time.time() - s)

        if round(100. * i / len(train_dataloader), 1) > cur_schedule:
            cur_schedule += .1
            if pms['rdd']:
                print(
                    f"Train epoch {epoch}:["
                    f"{str(i * pms['batch']).zfill(len((str(len(train_dataloader.dataset)))))}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.1f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.6f} |'
                    f'\tMSE loss 1: {out_criterion["mse_loss1"].item():.6f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.6f} |'
                    f'\tTime: {t.avg:.3f}'
                )
            else:
                print(
                    f"Train epoch {epoch}:["
                    f"{str(i * pms['batch']).zfill(len((str(len(train_dataloader.dataset)))))}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.1f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.6f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.6f} |'
                    f'\tTime: {t.avg:.3f}'
                )
            t.reset()

            if round(100. * i / len(train_dataloader), 1) > cur_save_schedule:
                cur_save_schedule += 10
                file_name = f"{pms['save_checkpoint']}/temp/{pms['lmbda']}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        'i': i,
                        "state_dict": model.state_dict()
                    },
                    file_name)
                print(f'Save temporary model as {file_name}')


def test_one_epoch(
        model,
        criterion,
        test_dataloader,
        epoch):
    model.eval()
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    if pms['rdd']:
        mse_loss1 = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            isi = d[0].to(pms['device'])
            img = d[1].to(pms['device'])
            out_net = model(isi)
            out_criterion = criterion(out_net, img)

            loss.update(out_criterion["loss"].item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            if pms['rdd']:
                mse_loss1.update(out_criterion["mse_loss1"].item())
            mse_loss.update(out_criterion["mse_loss"].item())

    if pms['rdd']:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.6f} |"
            f"\tMSE loss 1: {mse_loss1.avg:.6f} |"
            f"\tMSE loss: {mse_loss.avg:.6f} |"
            f"\tBpp loss: {bpp_loss.avg:.6f}"
        )
    else:
        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss.avg:.6f} |"
            f"\tMSE loss: {mse_loss.avg:.6f} |"
            f"\tBpp loss: {bpp_loss.avg:.6f}"
        )

    if os.path.exists(f"{pms['save_checkpoint']}/temp/{pms['lmbda']}.pth"):
        os.remove(f"{pms['save_checkpoint']}/temp/{pms['lmbda']}.pth")

    return loss.avg


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


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

    net = Joint_Net_with_conv_att(pms['load_checkpoint'], pms['device'])
    if pms['recon_detach']:
        for p in net.recon_net.parameters():
            p.requires_grad_(False)
    if pms['com_detach']:
        for p in net.com_net.parameters():
            p.requires_grad_(False)
        for p in net.first_encoder_layer.parameters():
            p.requires_grad_(False)
    net = net.to(pms['device'])
    # net = CustomDataParallel(net)
    optimizer = create_optimizers(net, pms['learning_rate'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=.5)
    if pms['rdd']:
        criterion = RDDLoss(pms['lmbda'])
    else:
        criterion = RateDistortionLoss(pms['lmbda'])

    if args.inference:
        print('Inferencing.')
        checkpoint = torch.load(pms['load_checkpoint'], map_location=pms['device'])
        net.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]
        test_one_epoch(
            net,
            criterion,
            test_dataloader,
            epoch)
        return

    last_epoch = 0
    best_loss = float("inf")
    best_iteration = 0
    # if pms['load_checkpoint']:
    #     try:
    #         checkpoint = torch.load(pms['load_checkpoint'], map_location=pms['device'])
    #         last_epoch = checkpoint["epoch"] + 1
    #         best_iteration = checkpoint["epoch"]
    #         best_loss = checkpoint["loss"]
    #         if os.path.exists(f"{pms['save_checkpoint']}/temp/{pms['lmbda']}.pth"):
    #             print("Loading from ", f"{pms['save_checkpoint']}/temp/{pms['lmbda']}.pth")
    #             net.load_state_dict(torch.load(f"{pms['save_checkpoint']}/temp/{pms['lmbda']}.pth", map_location=pms['device'])['state_dict'])
    #         else:
    #             print("Loading from ", pms['load_checkpoint'])
    #             net.load_state_dict(checkpoint["state_dict"])
    #         # optimizer.load_state_dict(checkpoint["optimizer"])
    #         # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    #         print("Loading finish")
    #     except FileNotFoundError:
    #         print("Loading failed")
    # else:
    #     print("Loading from two pre-trained models.")
    #     net.recon_net.load_state_dict(torch.load(pms['recon_net_path'], map_location=pms['device'])['state_dict'])
    #     net.com_net.load_state_dict(
    #         torch.load(f'/home/kxfeng/iccv/checkpoint/baseline/cheng2020_{pms["lmbda"]}.pth.tar',
    #                    map_location=pms['device'])['state_dict'])
    #     print("Loading finish")

    for epoch in range(last_epoch, pms['epochs']):
        print(f"Current best loss: {best_loss}, iteration: {best_iteration}.")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            pms['clip_max_norm']
        )
        loss = test_one_epoch(
            net,
            criterion,
            test_dataloader,
            epoch)
        lr_scheduler.step(loss)
        if loss < best_loss:
            best_loss = loss
            best_iteration = epoch
            if pms['save']:
                print(f'Save model as {pms["save_checkpoint"]}')
                file_name = f"{pms['save_checkpoint']}/joint-{pms['lmbda']}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "pms": pms
                    },
                    file_name)


parser = argparse.ArgumentParser(description='Reconstruction and Compression for spike sequences.')
parser.add_argument('--lmbda', type=int, required=False)
parser.add_argument('--device', type=str, required=False)
parser.add_argument('--batch_size', default=3, type=int)
parser.add_argument('--worker_num', default=2, type=int)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--recon_detach', action='store_true')
parser.add_argument('--com_detach', action='store_true')
parser.add_argument('--rdd', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    print(f'current pid: {os.getpid()}')
    pms['lmbda'] = args.lmbda
    pms['device'] = args.device
    pms['batch'] = args.batch_size
    pms['worker_num'] = args.worker_num
    pms['load_checkpoint'] = f'/home/kxfeng/iccv/checkpoint/rdd/joint-{args.lmbda}.pth'
    pms['rdd'] = args.rdd
    pms['recon_detach'] = args.recon_detach
    pms['com_detach'] = args.com_detach
    pms['save_checkpoint'] = '/home/kxfeng/iccv/checkpoint/rdd/conv_att'

    # pms['lmbda'] = 512
    # pms['device'] = 'cuda:0'
    # pms['batch'] = 16
    # pms['worker_num'] = 2
    # pms['load_checkpoint'] = f'/home/kxfeng/iccv/checkpoint/rdd/joint-{pms["lmbda"]}.pth'
    # pms['rdd'] = False
    # pms['recon_detach'] = True
    # pms['com_detach'] = True
    # pms['save_checkpoint'] = '/home/kxfeng/iccv/checkpoint/rdd/conv_att'
    print(json.dumps(pms, indent=2))
    main()
