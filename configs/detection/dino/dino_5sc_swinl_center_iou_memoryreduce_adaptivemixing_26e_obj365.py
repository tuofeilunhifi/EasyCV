_base_ = './dino_5sc_swinl_center_iou_memoryreduce_26e_obj365.py'

# model settings
model = dict(head=dict(transformer=dict(use_adaptivemixing=True)))
