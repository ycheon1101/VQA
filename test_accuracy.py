from utils import get_detection_info_test
from dataset import dataloader_test, classes
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/vqa_10000')
import pdb
import numpy as np

# param
box_threshold = 0.35
text_threshold = 0.25
count = 0
correct_predictions = 0
total_predictions = 0

for data in dataloader_test:
    prompt = []
    img, label = data
    label = classes[label]
    prompt.append(label)
    detections, detected_labels = get_detection_info_test(img, prompt, box_threshold, text_threshold)
    img = img.squeeze().permute(1, 2, 0).numpy()
    if len(detected_labels) == 0:
        writer.add_image(f'Image_{count}_not_matched', img, count, dataformats='HWC')
        writer.add_text(f'Label_{count}_not_matched', f'Predicted: {detected_labels}, Actual: {label}', count)
    elif detected_labels[0].split(' ')[0] == label:
        correct_predictions += 1
    total_predictions += 1

    if count % 50 == 0:
        writer.add_image(f'Image_{count}', img, count, dataformats='HWC')
        writer.add_text(f'Label_{count}', f'Predicted: {detected_labels}, Actual: {label}', count)

    accuracy = correct_predictions / total_predictions
    writer.add_scalar('Accuracy', accuracy, count)
    count += 1
writer.close()
    
