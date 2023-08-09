import numpy
import torch

from whole_net.joint_net import Joint_Net_LIC_with_lstm_att, Joint_Net_LIC_with_bilstm_att

if __name__ == '__main__':
    device = 'cuda:1'
    model = Joint_Net_LIC_with_bilstm_att(None, None)
    model = model.to(device)
    checkpoint_path = '/home/kxfeng/joint-0.05.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print('Loading finish.')

    spk_path = '/data2/kxfeng/vimeo_septuplet_spike/00001/0052.npy'
    spk = numpy.load(spk_path)
    spk = torch.from_numpy(spk).float()

    # spk_voxel = spk[0: 41, :, :]
    spk_voxel = spk[0: 41, :, 96: -96]
    spk_voxel = torch.unsqueeze(spk_voxel, 0)
    spk_voxel = spk_voxel.to(device)

    with torch.no_grad():
        att_map = model.temporal_att(spk_voxel)
        att_map = att_map.cpu().numpy()
        numpy.save('/home/kxfeng/soam.npy', att_map)

