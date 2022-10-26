_base_ = './detr_r50_8x2_150e_coco.py'

model = dict(
    head=dict(
        transformer=dict(
            _delete_=True,
            type='SlotTransformer',
            num_prototypes=100,
            in_channels=2048,
            d_model=256,
            temp=0.07)))

data = dict(imgs_per_gpu=64)
