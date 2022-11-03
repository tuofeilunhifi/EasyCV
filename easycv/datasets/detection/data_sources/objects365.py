import os.path as osp

from xtcocotools.coco import COCO

from easycv.datasets.detection.data_sources.coco import DetSourceCoco
from easycv.datasets.registry import DATASOURCES

objv2_ignore_list = [
    # images exist in annotations but not in image folder.
    'patch16/objects365_v2_00908726.jpg',
    'patch6/objects365_v1_00320532.jpg',
    'patch6/objects365_v1_00320534.jpg',
]


@DATASOURCES.register_module
class DetSourceObjects365(DetSourceCoco):
    """
    objects365 data source.

    The form of the objects365 dataset folder build:
        |- objects365
            |- annotation
                |- zhiyuan_objv2_train.json
                |- zhiyuan_objv2_val.json
            |- train
                |- patch0
                    |- *****(imageID)
                |- patch1
                    |- *****(imageID)
                ...
                |- patch50
                    |- *****(imageID)
            |- val
                |- patch0
                    |- *****(imageID)
                |- patch1
                    |- *****(imageID)
                ...
                |- patch43
                    |- *****(imageID)
    """

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.
        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]

            # rename filename and filter wrong data
            info['patch_name'] = osp.join(
                osp.split(osp.split(info['file_name'])[0])[-1],
                osp.split(info['file_name'])[-1])
            if info['patch_name'] in objv2_ignore_list:
                continue

            info['filename'] = info['patch_name']

            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
