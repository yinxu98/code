data = dict(
    type='FUSARShip',
    root='../data/fusarship',
    n_class=3,
    percentage=70,
    image_size=64,
    normalize=dict(
        mean=[0.05684700353123015, 0.05684700353123015, 0.05684700353123015],
        std=[0.11367373919156237, 0.11367373919156237, 0.11367373919156237],
    ),
    workers=2,
    batch_size=dict(pretrain=1024, train=512, val=2048),
)
