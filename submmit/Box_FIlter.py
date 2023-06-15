import numpy as np


def iou(box1, box2):
    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou



def box_filter(boxes, scores, labels , imgsz):
    limit_up = 0.4
    limit_down = 0.6
    iou_thr = 0.4

    boxes_re = []
    scores_re = []
    labels_re = []

    l2_boxes_re = []
    l2_scores_re = []
    l2_labels_re = []
    l2_ids = []
    l1_ids = []
    l0_ids = []

    del_limit_lists = []

    for i, box in enumerate(boxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        xm = (x1+x2)/2
        ym = (y1+y2)/2

        if limit_up < ym <limit_down:

            # boxes_re.append(box)
            # scores_re.append(scores[i])
            # labels_re.append(labels[i])

            if labels[i] == 2:
                l2_ids.append(i)
                # l2_boxes_re.append(box)
                # l2_scores_re.append(scores[i])
                # l2_labels_re.append(labels[i])

            elif labels[i] == 1:
                l1_ids.append(i)
                # l1_boxes_re.append(box)
                # l1_scores_re.append(scores[i])
                # l1_labels_re.append(labels[i])

            else:
                l0_ids.append(i)
                # l0_boxes_re.append(box)
                # l0_scores_re.append(scores[i])
                # l0_labels_re.append(labels[i])
        else:
            del_limit_lists.append(i)


    id_list = np.arange(0, len(labels)).tolist()

    if len(del_limit_lists) != 0:
        for del_limit in del_limit_lists:

            id_list.remove(del_limit)

    # if len(l2_ids)!=0 and len(l1_ids)!=0:
    #    for l2_id in l2_ids:
    #        for l1_id in l1_ids:
    #            l12_iou = iou(boxes[l2_id], boxes[l1_id])
    #            if l12_iou > iou_thr :
    #                if  boxes[l2_id][3] > boxes[l1_id][3] :
    #                       id_list.remove(l1_id)
    #                       l1_ids.remove(l1_id)
    #                else:
    #                    id_list.remove(l2_id)
    #                    l2_ids.remove(l2_id)
    #                    break

    boxes_re =  boxes[id_list]
    scores_re = scores[id_list]
    labels_re =  labels[id_list]

    return boxes_re,  scores_re, labels_re