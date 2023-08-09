pms = {
        'spk_path': '/home/kxfeng/vimeo_septuplet_spike',
        'gt_path': '/home/klin/data/vimeo_septuplet/sequences',
        'recon_net_path': '/home/kxfeng/iccv/checkpoint/recon_net.pth',
        'com_net_path': '/home/kxfeng/iccv/checkpoint/cheng2020_128.pth.tar',
        'epochs': 10000,
        'learning_rate': 1e-5,
        'worker_num': 1,
        'batch': 1,
        'test_batch': 1,
        'lmbda': 256,   # modify saving path
        'device': 'cuda:0',
        'save': True,
        'seed': None,
        'clip_max_norm': 1.,
        'load_checkpoint': '/home/kxfeng/iccv/checkpoint/recon_net.pth',
        'save_checkpoint': f'/home/kxfeng/iccv/checkpoint/joint',
        'log': './logs'
    }