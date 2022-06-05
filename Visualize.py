# Author: Fangdong Wu
# Time: 04.06.2021
# a visualization function to draw bounding boxes and id

import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def draw_bbox(img, tracks):
    for i in range(len(tracks)):
        if (tracks[i].is_confirmed()) & (tracks[i].hits > 2):
            temp_box = tracks[i].tlwh
            id = '%d' % tracks[i].track_id
            x = round(temp_box[0])
            y = round(temp_box[1])
            x_plus_w = round(temp_box[2] + x)
            y_plus_h = round(temp_box[3] + y)

            color = compute_color_for_labels(tracks[i].track_id)
            t_size = cv2.getTextSize(id, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 3)
            cv2.rectangle(img, (x, y), (x + t_size[0] + 3, y + t_size[1] + 4), color, -1)
            cv2.putText(img, id, (x, y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

def draw_num_obj(tracker, num_obj):
    num = 'Num: %d' % num_obj
    t_size = cv2.getTextSize(num, cv2.FONT_HERSHEY_PLAIN, 5, 5)[0]
    x = 10
    y = 10
    cv2.putText(tracker.frame, num, (x, y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 5, [0, 0, 255], 5)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)