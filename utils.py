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
import spacy
import webcolors

device = 'cuda:0'

# path
grounded_sam_path_ = str(grounded_sam_path)

# get grounding dino model
def get_grounding_dino_model(HOME):
    GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weight", "groundingdino_swint_ogc.pth")
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    return grounding_dino_model

# get sam model
def get_sam_model(HOME):
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = os.path.join(HOME, "weight", "sam_vit_h_4b8939.pth")
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

# strength prompt 
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

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

# get object detection info
def get_detection_info(src_img_path, img_name, prompt, box_threshold, text_threshold, save_img=True, img_save_path=None):
    # img
    src_img_path = f'{src_img_path}/{img_name}'
    image = cv2.imread(src_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get model
    grounding_dino_model = get_grounding_dino_model(grounded_sam_path_)

    # param
    CLASSES = prompt
    BOX_TRESHOLD = box_threshold
    TEXT_TRESHOLD = text_threshold

    detections = grounding_dino_model.predict_with_classes(
                    image=image,
                    # classes=enhance_class_name(class_names=CLASSES),
                    classes=CLASSES,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )
    
    labels = [
                f"{CLASSES[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections
            ]
    
    if save_img == True:
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        plt.imsave(f'{img_save_path}/{img_name[:-4]}_grounding_dino.png', annotated_frame)
    
    return detections, labels

def get_segment_info(src_img_path, img_name, detections, save_img=True, img_save_path=None):
    # get sam model
    sam_predictor = get_sam_model(grounded_sam_path_)
    # img
    src_img_path = f'{src_img_path}/{img_name}'
    image = cv2.imread(src_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections.mask = segment(
                            sam_predictor=sam_predictor,
                            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                            xyxy=detections.xyxy
                            )
    if save_img == True:
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        plt.imsave(f'{img_save_path}/{img_name[:-4]}_segmap.png', annotated_image)
    
    return detections

# get prompt and the question type
def get_question_prompt_info():
    # model
    nlp = spacy.load("en_core_web_sm")

    # sentence
    sentence = input('input sentence: ')
    sentence = sentence.lower()

    # instantiation
    doc = nlp(sentence)

    # initialize
    noun_phrases = []
    question_type = -1

    # check list
    check_list = ['there', 'made of', 'color', 'how']

    # check question type
    for idx, check in enumerate(check_list):
        if check in sentence:
            question_type = idx 

    # extract noun
    for chunk in doc.noun_chunks:
        noun_phrase = ' '.join([token.text for token in chunk if token.pos_ != 'DET'])
        noun_phrases.append(noun_phrase)

    return sentence, question_type, noun_phrases

# calculate iou
def calculate_iou(box1, box2):
    # top-left, bottom-right
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # intersection coords
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    # intersection
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # bb
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

    # iou
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

# color
def get_rgb_values(image_path, mask):
    # img
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # coords where the mask == True
    coords = np.where(mask)

    # r, g, b value
    r_values = image_np[coords[1], coords[2], 0]
    g_values = image_np[coords[1], coords[2], 1]
    b_values = image_np[coords[1], coords[2], 2]

    avg_r = np.mean(r_values)
    avg_g = np.mean(g_values)
    avg_b = np.mean(b_values)

    return int(avg_r), int(avg_g), int(avg_b)

def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_color):
    try:
        color_name = webcolors.rgb_to_name(rgb_color)
    except ValueError:
        color_name = closest_color(rgb_color)
    return color_name

# get object detection info for test
def get_detection_info_test(img_tensor, prompt, box_threshold, text_threshold):
    # img
    image = img_tensor.squeeze().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    # get model
    grounding_dino_model = get_grounding_dino_model(grounded_sam_path_)

    # param
    CLASSES = prompt
    BOX_TRESHOLD = box_threshold
    TEXT_TRESHOLD = text_threshold

    detections = grounding_dino_model.predict_with_classes(
                    image=image,
                    # classes=enhance_class_name(class_names=CLASSES),
                    classes=CLASSES,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD
                )
    try:
        labels = [
                f"{CLASSES[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections
            ]
    except:
        labels = [
                    f"{CLASSES[class_id]} {confidence:0.2f}" if class_id is not None and type(class_id) == int else 'unknown'
                    for _, _, confidence, class_id, _
                    in detections
                ]
    
    # if save_img == True:
    #     # annotate image with detections
    #     box_annotator = sv.BoxAnnotator()
    #     annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        # plt.imsave(f'{img_save_path}/{img_name[:-4]}_grounding_dino.png', annotated_frame)
    
    return detections, labels









