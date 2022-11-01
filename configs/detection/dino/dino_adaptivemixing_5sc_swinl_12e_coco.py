_base_ = [
    './dino_5sc_swinl.py', '../common/dataset/autoaug_coco_detection.py',
    './dino_schedule_1x.py'
]

# model settings
model = dict(head=dict(transformer=dict(use_adaptivemixing=True)))
