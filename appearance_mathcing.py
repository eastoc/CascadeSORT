# Author: Fangdong Wu
# Time: 06.22.2021
# Cascade Matching

import numpy as np
import kalman_filter as kf
import feature_extractor
import iou_matching as im

def ensemble_mathcing(tracks, candidates, iou_gate_matrix):
    """
    Inputs tracks and detections，outputs cascade gated matrix via cascade matching
    Mahalanobis distance and VLAD cascade matching for confirmed track, IoU matching for tentative track
    :param tracks:
    :param candidates:
    :param iou_gate_matrix:
    :return: cascade_cost_matrix
    """
    num_tracks = len(tracks)
    num_candidates = len(candidates)
    cascade_cost_matrix = np.zeros(iou_gate_matrix.shape)
    KF = kf.KalmanTracker()

    maha_threshold = 9.4877
    VLAD_threshold = 0.8
    is_match = 0 # Cascade matching flag:1 represents matching，0 is un-matching
    is_iou_match = 0 # IoU matching flag
    iou_threshold = 0.3
    iou_flag = -1
    matched_detection = np.zeros(num_candidates)# 用来检查 detection 是否被匹配，避免被多个 tracks 争抢
    # detections优先被tentative的tracks匹配
    vec_one = np.ones(num_candidates)
    #matched_detection = np.dot(iou_gate_matrix, vec_one)

    for i in range(num_tracks):
        vec_vlad_score = np.zeros(iou_gate_matrix.shape[1])
        for j in range(num_candidates):

            if tracks[i].is_confirmed():
                # Compute Mahalanobis distance
                iou = im.compute_iou(tracks[i].tlwh, candidates[j].tlwh, iou_threshold)
                measurement = im.tlwh2kalman_measurement(candidates[j].tlwh)
                maha = KF.gating_distance(tracks[i].mean, tracks[i].covariance, measurement)
                #print('maha is ',maha)
                if maha < maha_threshold:
                    # 马氏距离小于阈值，计算并判断simi，如果simi小于阈值
                    # Compute similarity using VLAD
                    try:
                        similarity = feature_extractor.VLAD(tracks[i].img_crop, candidates[j].img_crop, SeedNum=10)
                        print(similarity)
                    except:
                        print("The error occurred in VLAD")
                    if similarity >= VLAD_threshold:
                        #print('VLAD')
                        vec_vlad_score[j] = similarity
                        is_match = 1

        if is_match == 1:
            flag = max_flag(vec_vlad_score)
            cascade_cost_matrix[i][flag] = 1
            is_match = 0
    vec_one_tracks = np.ones([num_candidates, 1])
    vec_one_detections = np.ones([num_tracks, 1])
    vec_tracks = np.dot(cascade_cost_matrix, vec_one_tracks)
    vec_detections = np.dot(cascade_cost_matrix.T, vec_one_detections)
    unmatched_tracks = []
    unmatched_detections = []
    num_unmatched_tracks = 0
    num_unmatched_detections = 0
    for i in range(num_tracks):
        if vec_tracks[i] == 0:
            unmatched_tracks.append(i)
            num_unmatched_tracks += 1
    for i in range(num_candidates):
        if vec_detections[i] == 0:
            unmatched_detections.append(i)
            num_unmatched_detections += 1

    if num_unmatched_detections != 0:
        for i in range(num_unmatched_tracks):
            index_i = unmatched_tracks[i]
            if tracks[index_i].is_confirmed():
                for j in range(num_unmatched_detections):
                    index_j = unmatched_detections[j]
                    if iou_gate_matrix[index_i, index_j] == 1:
                        cascade_cost_matrix[index_i, index_j] = 1

    for i in range(num_tracks):
        if tracks[i].is_tentative():
            for j in range(num_candidates):
                cascade_cost_matrix[i, j] = iou_gate_matrix[i, j]


    return cascade_cost_matrix

def cascade_iou_metric(tracks,candidates):
    """
    IoU metric in cascade matching
    :param tracks:
    :param candidates:
    :return:
    """
    iou_threshold = 0.3
    num_tracks = len(tracks)
    num_candidates = len(candidates)
    iou_matrix = np.zeros([num_tracks, num_candidates])

    for i in range(num_tracks):
        max_iou = 0
        max_iou_flag = None
        if tracks[i].is_confirmed():  # If confirmed state，compute IoU
            for j in range(num_candidates):
                iou_matrix[i][j] = im.compute_iou(tracks[i].tlwh, candidates[j].tlwh, iou_threshold)
    return iou_matrix

def max_flag(vector):
    """
    Find the index of maxmium
    :param vector: ndarray
    :return: The index of max number in a vector
    """
    max = 0
    flag = None
    for i in range(len(vector)):
        if vector[i] > max:
            flag = i
            max = vector[flag]
    return flag