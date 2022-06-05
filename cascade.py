import iou_matching
import numpy as np
import track
import kalman_filter as kf
import iou_matching as im
import appearance_mathcing
from scipy.optimize import linear_sum_assignment

def match(tracks, candidates, num_obj, KF):
    """
    New match function
    :param tracks:
    :param candidates:
    :param num_obj:
    :return: the number of objects can be confirmed
    """
    num_tracks = len(tracks)
    num_candidates = len(candidates)

    cascade_cost_matrix = cascade_matching(tracks, candidates)
    tracks_index, det_index = linear_sum_assignment(cascade_cost_matrix, True)
    matched_tracks_vector = np.zeros(num_tracks)
    matched_candidates_vector = np.zeros(num_candidates)
    # KF = kf.KalmanTracker()
    # Update matched tracks' state
    for i, x in enumerate(tracks_index):
        y = det_index[i]
        if cascade_cost_matrix[x, y] == 1:
            kf.update_velo(tracks[x], candidates[y])
            tracks[x].mean, tracks[x].covariance = KF.update(tracks[x].mean,
                                                         tracks[x].covariance,
                                                         iou_matching.tlwh2kalman_measurement(candidates[y].tlwh))
            tracks[x].update_matched(candidates[y])

            # if hits >= 3, switch state from tentative to confirmed
            if (tracks[x].is_tentative()) & (tracks[x].hits == tracks[x]._n_init):
                tracks[x].state = track.TrackState.Confirmed
                num_obj += 1
                tracks[x].track_id = num_obj

            matched_tracks_vector[x] = 1
            matched_candidates_vector[y] = 1

    # Update unmatched track state,
    # If time_since_update < max age, kalman filter predict track's state, else delete it

    for i in range(num_tracks):
        if matched_tracks_vector[i] == 0:
            if (tracks[i].time_since_update < tracks[i]._max_age) & (tracks[i].is_confirmed()):
                tracks[i].update_unmatched()

                # Kalman filter predict
                tracks[i].mean, tracks[i].covariance = KF.predict(tracks[i].mean, tracks[i].covariance)
                tracks[i].update_tlwh()

            elif (tracks[i].time_since_update == tracks[i]._max_age) & (tracks[i].is_confirmed()):
                tracks[i].state = track.TrackState.Deleted

    # Create new tracks in term of unmatched detections
    for i in range(num_candidates):
        if matched_candidates_vector[i] == 0:
            track_temp = track.Track(candidates[i].tlwh, candidates[i].img_crop,
                                     mean=None, covariance=None, track_id=None,
                                     n_init=3, max_age=20)
            track_temp.mean, track_temp.covariance = KF.initiate(im.tlwh2kalman_measurement(track_temp.tlwh))
            tracks.append(track_temp)
            track_temp = []
            num_tracks = len(tracks)

    return num_obj


def cascade_matching(tracks, candidates):
    """
    process cascade_cost_matrix, output matched tracks and unmatched detections
    :param tracks: Track class
    :param candidates: Candidates class
    :return: ndarray
        cost matrix of cascade matching
    """
    iou_cost_matrix = iou_matching.compute_iou_matrix(tracks, candidates)
    iou_gate_matrix = np.zeros(iou_cost_matrix.shape)
    #cascade_cost_matrix = np.zeros(iou_cost_matrix.shape)

    # IoU gate matching
    for i in range(iou_cost_matrix.shape[0]):
        for j in range(iou_cost_matrix.shape[1]):
            if iou_cost_matrix[i][j] != 0:
                iou_gate_matrix[i][j] = 1
    # cascade matching
    cascade_cost_matrix = appearance_mathcing.ensemble_mathcing(tracks, candidates, iou_gate_matrix)


    return cascade_cost_matrix

