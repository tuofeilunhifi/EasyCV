_base_ = './dino_5sc_swinl_center_iou_memoryreduce_18e_obj2coco.py'

# model settings
model = dict(head=dict(transformer=dict(use_adaptivemixing=True)))
