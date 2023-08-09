pms = {
        # 'spk_path': '/home/kxfeng/vimeo_septuplet_spike',
        # 'gt_path': '/home/klin/data/vimeo_septuplet/sequences',
        'spk_path': '/data/kxfeng/vimeo_septuplet_spike',
        'gt_path': '/data/klin/vimeo_septuplet/sequences',

        # 'spk_path': '/backup4/kxfeng/vimeo_septuplet_spike',
        # 'gt_path': '/backup2/whduan/vimeo_septuplet/sequences',

        # 'spk_path': '/backup2/kxfeng/vimeo_septuplet_spike',
        # 'gt_path': '/data/klin/vimeo_septuplet/sequences',

        'epochs': 1000,
        'learning_rate': 1e-6,
        'worker_num': 2,
        'batch': 2,
        'test_batch': 8,
        'lmbda': None,
        'device': None,
        'save': True,
        'seed': None,
        'clip_max_norm': 1.,
        'load_checkpoint': None,
        'save_checkpoint': '/home/kxfeng/iccv/checkpoint',
        'log': './logs'
    }