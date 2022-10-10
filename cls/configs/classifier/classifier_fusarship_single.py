_base_ = ['./classifier_fusarship_test.py']

checkpoints = dict(
    folders=[
        '../work_dirs/untrained',
        '../work_dirs/multiembedding_fusarship_pretrain',
    ],
    index=[
        'res50_0020.pth',
    ],
)
