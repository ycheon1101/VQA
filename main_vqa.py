# python3 -m spacy download en_core_web_sm

from utils import get_detection_info, get_segment_info, get_question_prompt_info, \
                    calculate_iou, get_rgb_values, get_color_name
import pdb

# param
src_img_path = './img/src'
save_img_path = './img/result'
box_threshold = 0.35
text_threshold = 0.25
img_name = 'image4.png'


def vqa():
    answer = ''
    question, question_type, noun = get_question_prompt_info()
    # question check
    check = False
    while check == False:
        print(f'Do you want to detect {noun}?\nAfter "detect" should be "a word". \nIf it is not a word, plz type "No" and input only one word.')
        temp = input('Yes or No? ')
        temp = temp.lower()
        check = True if temp == 'yes' else False
        if check == False:
            print(f'Please input simply.')
            _, _, noun = get_question_prompt_info()
    # is / are there ~
    if question_type == 0:
        prompt = noun
        detections, labels = get_detection_info(src_img_path, img_name, prompt, box_threshold, text_threshold, save_img=True, img_save_path=save_img_path)
        if len(labels) >= 1:
            label = labels[0].split(' ')[0]
        elif len(labels) == 0:
            label = None
        sam_detections = get_segment_info(src_img_path, img_name, detections, save_img=True, img_save_path=save_img_path)
        return 'Yes' if label in noun else 'No', question
    # is ~ made of ~?
    elif question_type == 1:
        src_prompt = [noun[0]]
        target_prompt = [noun[1]]
        src_detections, _ = get_detection_info(src_img_path, img_name, src_prompt, box_threshold, text_threshold, save_img=True, img_save_path=save_img_path)
        target_detections, _ = get_detection_info(src_img_path, img_name, target_prompt, box_threshold, text_threshold, save_img=True, img_save_path=save_img_path)
        # when src prompt or target prompt is not detected in the img
        if len(src_detections.xyxy) == 0 or len(target_detections.xyxy) == 0:
            return 'No', question
        src_sam_detections = get_segment_info(src_img_path, img_name, src_detections, save_img=True, img_save_path=save_img_path)
        target_sam_detections = get_segment_info(src_img_path, img_name, target_detections, save_img=True, img_save_path=save_img_path)
        iou = calculate_iou(src_sam_detections.xyxy[0], target_sam_detections.xyxy[0])
        # threshold: 0.2
        return 'Yes' if iou >= 0.2 else 'No', question
    # what is the color ~?
    elif question_type == 2:
        prompt = noun
        detections, labels = get_detection_info(src_img_path, img_name, prompt, box_threshold, text_threshold, save_img=True, img_save_path=save_img_path)
        if len(labels) >= 1:
            label = labels[0].split(' ')[0]
        elif len(labels) == 0:
            label = None
        sam_detections = get_segment_info(src_img_path, img_name, detections, save_img=True, img_save_path=save_img_path)
        if label in noun:
            src_img = f'{src_img_path}/{img_name}'
            avg_r, avg_g, avg_b = get_rgb_values(src_img, sam_detections.mask)
            color_name = get_color_name((avg_r, avg_g, avg_b))
            return f'\nAverage RGB Values is R={avg_r}, G={avg_g}, B={avg_b}\
                    \nColor Name: {color_name}\
                    \nIf you want to check the r, g, b color, plz visit "https://rgb.to/"', question
        else:
            return f'There is no {noun[0]} in the image.', question
    # how many ~
    elif question_type == 3:
        count = 0
        prompt = noun
        detections, labels = get_detection_info(src_img_path, img_name, prompt, box_threshold, text_threshold, save_img=True, img_save_path=save_img_path)
        count = len(labels)
        sam_detections = get_segment_info(src_img_path, img_name, detections, save_img=True, img_save_path=save_img_path)
        return f'The number of {noun[0]} in this image is {count}.', question



    
if __name__ == '__main__':
    answer, question = vqa()
    print(f'Question: {question}')
    print(f'Answer: {answer}')


