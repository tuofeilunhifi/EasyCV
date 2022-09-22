_base_ = [
    './dino_4sc_r50.py', '../common/dataset/autoaug_coco_detection.py',
    './dino_schedule_1x.py'
]

# model settings
model = dict(
    head=dict(
        use_centerness=True,
        use_iouaware=True,
        use_tokenlabel=True,
        losses_list=['labels', 'boxes', 'centerness', 'iouaware'],
        transformer=dict(multi_encoder_memory=True),
        weight_dict=dict(loss_ce=2, loss_center=2, loss_iouaware=2, loss_token_focal=2, loss_token_dice=2)))
