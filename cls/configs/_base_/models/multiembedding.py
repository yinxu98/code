# __vector_dim = 512  # res18,34
__vector_dim = 2048 # res50

model = dict(
    type='MultiEmbedding',
    backbone=dict(
        type='MyResNetV1d',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
    ),
    n_backbone=2,
    predictor=dict(
        dim_in=__vector_dim,
        dim_mid=__vector_dim // 2,
        dim_out=__vector_dim,
    ),
    n_predictor=2,
)
