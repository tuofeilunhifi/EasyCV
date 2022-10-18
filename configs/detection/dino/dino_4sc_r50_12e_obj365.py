_base_ = [
    './dino_4sc_r50.py', '../common/dataset/autoaug_objects365_detection.py',
    './dino_schedule_1x.py'
]

model = dict(
    head=dict(dn_components=dict(dn_labelbook_size=365), num_classes=365))
