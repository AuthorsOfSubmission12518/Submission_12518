pms = {
        # 'spk_path': '/backup2/kxfeng/vimeo_septuplet_spike',
        # 'gt_path': '/data/klin/vimeo_septuplet/sequences',
        'spk_path': '/backup4/kxfeng/vimeo_septuplet_spike',
        'gt_path': '/backup2/whduan/vimeo_septuplet/sequences',
        'epochs': 10000,
        'learning_rate': 1e-5,
        'worker_num': 4,
        'batch': 1,
        'test_batch': 1,
        'device': 'cuda:0',
        'save': True,
        'seed': None,
        'clip_max_norm': 1.,
        'load_checkpoint': '/home/kxfeng/iccv/checkpoint/recon_net.pth',
        'save_checkpoint': None,
        'log': './logs'
    }