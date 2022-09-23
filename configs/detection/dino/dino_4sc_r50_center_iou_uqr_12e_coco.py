_base_ = [
    './dino_4sc_r50.py', '../common/dataset/autoaug_coco_detection.py',
    './dino_schedule_1x.py'
]

# model settings
model = dict(
    head=dict(
        use_centerness=True,
        use_iouaware=True,
        use_vector=True,
        processor_dct=dict(n_keep=256, gt_mask_len=128),
        losses_list=['labels', 'boxes', 'centerness', 'iouaware', 'masks'],
        weight_dict=dict(loss_ce=2, loss_center=2, loss_iouaware=2, loss_vector=1)))