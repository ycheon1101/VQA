import torch
import sys
import os
from pathlib import Path
import pdb

vqa_path = Path(os.getcwd())
root_path = vqa_path.parent
segmentation_path = root_path / 'segmentation'
grounded_sam_path = segmentation_path / 'grounded_sam'
grounding_dino_path = grounded_sam_path / 'GroundingDINO'
sys.path.append(str(grounding_dino_path))
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import pandas as pd
from PIL import Image

device = 'cuda:0'

# path
HOME = str(grounded_sam_path)
src_img_path = vqa_path / 'img' / 'src'
save_img_path = vqa_path / 'img' / 'result'
img_name = 'image2.png'
# pdb.set_trace()

# grounding DINO
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weight", "groundingdino_swint_ogc.pth")
# GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weight", "grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth")
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)


# SAM
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weight", "sam_vit_h_4b8939.pth")
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=device)
sam_predictor = SamPredictor(sam)


# mask annotation
SOURCE_IMAGE_PATH = f"{src_img_path / img_name}"

# CLASSES = ['banana']
CLASSES = ['cat']
# CLASSES = ['car']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# strength prompt 
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]
# pdb.set_trace()

# segment
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

# load image
image = cv2.imread(SOURCE_IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# gt_label_img = Image.open(gt_label_img)
# convert = ConvertGtToUsed()
# gt_label_img = convert(gt_label_img)
# # pdb.set_trace()

# detect objects
detections = grounding_dino_model.predict_with_classes(
image=image,
# classes=enhance_class_name(class_names=CLASSES),
classes=CLASSES,
box_threshold=BOX_TRESHOLD,
text_threshold=TEXT_TRESHOLD
)

# import pdb
# pdb.set_trace()

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
# f"{CLASSES[class_id]} {confidence:0.2f}" if class_id is not None and type(class_id) == int else 'unknown'
f"{CLASSES[class_id]} {confidence:0.2f}"
for _, _, confidence, class_id, _
in detections]
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
# pdb.set_trace()


# convert detections to masks
detections.mask = segment(
sam_predictor=sam_predictor,
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
xyxy=detections.xyxy
)
# pdb.set_trace()

# annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
# f"{CLASSES[class_id]} {confidence:0.2f}" if class_id is not None and type(class_id) == int else 'unknown'
f"{CLASSES[class_id]} {confidence:0.2f}" 
for _, _, confidence, class_id, _
in detections]
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# save image
plt.imsave(f'{save_img_path}/{img_name[:-4]}_grounding_dino.png', annotated_frame)
plt.imsave(f'{save_img_path}/{img_name[:-4]}_segmap.png', annotated_image)

pdb.set_trace()
