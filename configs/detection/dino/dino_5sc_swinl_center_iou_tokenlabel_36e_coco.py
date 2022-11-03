_base_ = './dino_5sc_swinl_12e_coco.py'

# learning policy
lr_config = dict(policy='step', step=[27, 33])

total_epochs = 36

# model settings
model = dict(
    head=dict(
        use_centerness=True,
        use_iouaware=True,
        use_tokenlabel=True,
        losses_list=['labels', 'boxes', 'centerness', 'iouaware'],
        weight_dict=dict(loss_ce=2, loss_center=2, loss_iouaware=2, loss_token_focal=2, loss_token_dice=2)))