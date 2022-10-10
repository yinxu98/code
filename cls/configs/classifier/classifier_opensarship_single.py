_base_ = ['./classifier_opensarship_test.py']

checkpoints = dict(
    folders=[
        # '../work_dirs/untrained',
        '../work_dirs/multiembedding_opensarship_pretrain',
    ],
    index=[
        # 'ResNetV1d18.pth',
        # 'ResNetV1d34.pth',
        'res34_0020.pth'
    ],
)
