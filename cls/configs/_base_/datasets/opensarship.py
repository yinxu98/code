data = dict(
    type='OpenSARShip',
    root='../data/opensarship',
    n_class=5,
    percentage=70,
    image_size=64,
    normalize=dict(
        mean=[
            0.026807075832908405, 0.026807075832908405, 0.026807075832908405
        ],
        std=[0.039102898421077066, 0.039102898421077066, 0.039102898421077066],
    ),
    workers=2,
    batch_size=dict(pretrain=1024, train=512, val=2048),
)
