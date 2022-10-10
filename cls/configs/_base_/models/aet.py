__vector_dim = 2048
model = dict(
    type='AET',
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
    aet_head=dict(
        dim_in=2 * __vector_dim,
        dim_mid=__vector_dim,
        dim_out=9,
    ),
)
