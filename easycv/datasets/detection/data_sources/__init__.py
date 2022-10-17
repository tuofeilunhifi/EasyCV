# Copyright (c) Alibaba, Inc. and its affiliates.
from .coco import DetSourceCoco
from .coco_panoptic import DetSourceCocoPanoptic
from .objects365 import DetSourceObjects365
from .pai_format import DetSourcePAI
from .raw import DetSourceRaw
from .voc import DetSourceVOC

__all__ = [
    'DetSourceCoco', 'DetSourceCocoPanoptic', 'DetSourcePAI', 'DetSourceRaw',
    'DetSourceVOC'
]
