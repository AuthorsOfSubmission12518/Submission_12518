pms = {
        # 'spk_path': '/home/kxfeng/vimeo_septuplet_spike',
        # 'gt_path': '/home/klin/data/vimeo_septuplet/sequences',
        # 'recon_net_path': '/home/kxfeng/iccv/checkpoint/recon_net_small.pth',
        # 'com_net_path': '/home/kxfeng/iccv/checkpoint/cheng2020_128.pth.tar',

        # 'spk_path': '/backup4/kxfeng/vimeo_septuplet_spike',
        # 'gt_path': '/backup2/whduan/vimeo_septuplet/sequences',

        'spk_path': '/backup2/kxfeng/vimeo_septuplet_spike',
        'gt_path': '/data/klin/vimeo_septuplet/sequences',
        'recon_net_path': '/home/kxfeng/iccv/checkpoint/recon_net_small.pth',
        'com_net_path': '/home/kxfeng/iccv/checkpoint/cheng2020_128.pth.tar',

        'epochs': 10000,
        'learning_rate': 1e-4,
        'worker_num': 1,
        'batch': 1,
        'test_batch': 1,
        'lmbda': 256,
        'device': 'cuda:1',
        'save': True,
        'seed': None,
        'clip_max_norm': 1.,
        'load_checkpoint': None,
        'save_checkpoint': '/home/kxfeng/iccv/checkpoint/',
        'log': './logs'
    }