_base_ = [
    './dino_4sc_swinl.py', '../common/dataset/autoaug_coco_detection.py',
    './dino_schedule_1x.py'
]

data = dict(imgs_per_gpu=4)

# optimizer_config = dict(update_interval=2)

# model settings
model = dict(head=dict(transformer=dict(use_adaptivemixing=True)))
