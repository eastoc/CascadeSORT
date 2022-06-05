# Written by Fangdong Wu
# Time: 04.04.2021
# A package is related to IoU

import numpy as np

def compute_iou(tlwh1, tlwh2, iou_threshold):
    """
    :param bbox1{x0,y0,x1,y1}
            {top-left,bottom-right}
    :param bbox2{x0,y0,x1,y1}
            {top-left,bottom-right}
    :return iou: IoU with two bboxes
    """
    sum_area = tlwh1[2] * tlwh1[3] + tlwh2[2] * tlwh2[3]
    tlbr1 = tlwh2tlbr(tlwh1)
    tlbr2 = tlwh2tlbr(tlwh2)

    # find the intersected rectangle
    left_line = max(tlbr1[0], tlbr2[0])
    top_line = max(tlbr1[1], tlbr2[1])
    right_line = min(tlbr1[2], tlbr2[2])
    bottom_line = min(tlbr1[3], tlbr2[3])

    if (right_line > left_line) & (bottom_line > top_line): #剔除框不相交的情况
        # compute the intersected area
        intersect_area = (right_line - left_line) * (bottom_line - top_line)
        # compute the union area
        union_area = sum_area - intersect_area
        #compute iou area
        iou = intersect_area/union_area
    else:
        iou = 0

    if (iou > iou_threshold) & (iou <= 1):
        return iou
    else:
        iou = 0
        return iou

def tlwh2tlbr(tlwh):
    """
    transform {top-left,width,height} to {top-left,bottom-right}
    :param tlwh:a data representive on bbox,
                {top-left,width,height}
    :return tlbr:a data representive on bbox,
                {top-left,bottom-right}
    """
    tlbr = [0, 0, 0, 0]
    tlbr[0] = tlwh[0]
    tlbr[1] = tlwh[1]
    tlbr[2] = tlwh[0] + tlwh[2]
    tlbr[3] = tlwh[1] + tlwh[3]
    return tlbr

def tlbr2tlwh(tlbr):
    """
    transform {top-left,bottom-right} to {top-left,width,height}
    :param tlbr: a data representive on bbox,
                {top-left,bottom-right}
    :return: tlwh:a data representive on bbox,
                {top-left,width,height}
    """
    tlwh = [0,0,0,0]
    tlwh[0] = tlbr[0]   # x-location of the top-left point
    tlwh[1] = tlbr[1]   # y-location of the top-lef point
    tlwh[2] = tlbr[2] - tlbr[0] # width
    tlwh[3] = tlbr[3] - tlbr[1] # height
    return tlwh

def tlwh2center(tlwh):
    """
    transform {top-left, width, height} to {center point, width, height}
    :param tlwh: ndarray
        {top-left, width, height}
    :return: center: ndarray
        {center point, width, height}
    """
    center = np.zeros([4, 1])
    center[0] = tlwh[0] + 1/2 * tlwh[2]
    center[1] = tlwh[1] + 1/2 * tlwh[3]
    center[2] = tlwh[2]
    center[3] = tlwh[3]
    return center

def tlwh2kalman_measurement(tlwh):
    """
    :param tlwh: bbox {top-left,width,height}
    :return: measurement: Kalman state vector is {x,y,ratio,height,x_vel,y_vel,ratio_vel,height_vel}
        note: "vel" is velocity, x_vel is velocity of x
    """
    measurement = np.zeros_like(tlwh)
    measurement[0] = tlwh[0] + 1/2 * tlwh[2]
    measurement[1] = tlwh[1] + 1/2 * tlwh[3]
    measurement[2] = tlwh[3]/tlwh[2]
    measurement[3] = tlwh[3]
    return measurement

def compute_iou_matrix(tracks,candidates):
    """
    Compute IoU cost matrix for tentative track
    :param tracks:
    :param candidates: detected bounding boxes
    :return: IoU cost matrix, a matrix sized {tracks' number} X {candidates' number}
    """

    iou_threshold = 0.25
    num_tracks = len(tracks)
    num_candidates = len(candidates)
    iou_matrix = np.zeros([num_tracks, num_candidates])

    for i in range(num_tracks):
        if tracks[i].is_deleted() == 0: # If confirmed or tentative，compute IoU
            for j in range(num_candidates):
                iou_matrix[i][j] = compute_iou(tracks[i].tlwh, candidates[j].tlwh, iou_threshold)

    return iou_matrix