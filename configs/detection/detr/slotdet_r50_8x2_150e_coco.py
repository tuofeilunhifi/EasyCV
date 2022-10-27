_base_ = './detr_r50_8x2_150e_coco.py'

model = dict(head=dict(transformer=dict(type='SlotTransformer')))

data = dict(imgs_per_gpu=16)
