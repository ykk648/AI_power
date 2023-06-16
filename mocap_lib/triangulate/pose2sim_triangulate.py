import os
import glob
import fnmatch
import numpy as np
import json
import itertools as it
import pandas as pd
import toml
from tqdm import tqdm
from scipy import interpolate
import logging
from pathlib import Path

from mocap_lib.triangulate.utils.common import computeP, weighted_triangulation, reprojection, \
    euclidean_distance, natural_sort


def zup2yup(Q):
    """
    Turns Z-up system coordinates into Y-up coordinates

    INPUT:
    - Q: pandas dataframe
    N 3D points as columns, ie 3*N columns in Z-up system coordinates
    and frame number as rows

    OUTPUT:
    - Q: pandas dataframe with N 3D points in Y-up system coordinates
    """

    # X->Y, Y->Z, Z->X
    cols = list(Q.columns)
    cols = np.array([[cols[i * 3 + 1], cols[i * 3 + 2], cols[i * 3]] for i in range(int(len(cols) / 3))]).flatten()
    Q = Q[cols]

    return Q


def interpolate_zeros(col, *kind):
    '''
    Interpolate missing points (of value 0.)

    INPUTS:
    - col: pandas column of coordinates
    - kind: 'linear', 'slinear', 'quadratic', 'cubic'. Default: 'cubic'

    OUTPUT:
    - col_interp: interpolated pandas column
    '''

    idx = col.index
    idx_good = np.where(np.isfinite(col))[0]  # index of non zeros
    if len(idx_good) <= 10: return col

    if not kind:  # 'linear', 'slinear', 'quadratic', 'cubic'
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind="cubic", bounds_error=False)
    else:
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind[0], bounds_error=False)
    col_interp = np.where(np.isfinite(col), col, f_interp(idx))  # replace nans with interpolated values
    col_interp = np.where(np.isfinite(col_interp), col_interp, np.nanmean(col_interp))  # replace remaining nans

    return col_interp


def recap_triangulate(config, error, nb_cams_excluded, keypoints_names, trc_path):
    '''
    Print a message giving statistics on reprojection errors (in pixel and in m)
    as well as the number of cameras that had to be excluded to reach threshold 
    conditions. Also stored in User/logs.txt.

    INPUT:
    - a Config.toml file
    - error: dataframe 
    - nb_cams_excluded: dataframe
    - keypoints_names: list of strings

    OUTPUT:
    - Message in console
    '''

    # Read config
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    error_threshold_triangulation = config.get('3d-triangulation').get('error_threshold_triangulation')
    likelihood_threshold = config.get('3d-triangulation').get('likelihood_threshold')

    # Recap
    calib = toml.load(calib_file)
    calib_cam1 = calib[list(calib.keys())[0]]
    fm = calib_cam1['matrix'][0][0]
    Dm = euclidean_distance(calib_cam1['translation'], [0, 0, 0])

    for idx, name in enumerate(keypoints_names):
        mean_error_keypoint_px = np.around(error.iloc[:, idx].mean(), decimals=1)  # RMS à la place?
        mean_error_keypoint_m = np.around(mean_error_keypoint_px * Dm / fm, decimals=3)
        mean_cam_excluded_keypoint = np.around(nb_cams_excluded.iloc[:, idx].mean(), decimals=2)
        logging.info(
            f'Mean reprojection error for {name} is {mean_error_keypoint_px} px (~ {mean_error_keypoint_m} m), reached with {mean_cam_excluded_keypoint} excluded cameras. ')

    mean_error_px = np.around(error['mean'].mean(), decimals=1)
    mean_error_mm = np.around(mean_error_px * Dm / fm * 1000, decimals=1)
    mean_cam_excluded = np.around(nb_cams_excluded['mean'].mean(), decimals=2)

    logging.info(
        f'--> Mean reprojection error for all points on all frames is {mean_error_px} px, which roughly corresponds to {mean_error_mm} mm. ')
    logging.info(
        f'\nCameras were excluded if likelihood was below {likelihood_threshold} and if the reprojection error was above {error_threshold_triangulation} px.')
    logging.info(f'In average, {mean_cam_excluded} cameras had to be excluded to reach these thresholds.')
    logging.info(f'\n3D coordinates are stored at {trc_path}.')


def triangulation_from_best_cameras(config, coords_2D_kpt, projection_matrices):
    '''
    Triangulates 2D keypoint coordinates, only choosing the cameras for which 
    reprojection error is under threshold.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: 
    - projection_matrices: list of arrays

    OUTPUTS:
    - Q: array of triangulated point (x,y,z,1.)
    - error_min: float
    - nb_cams_excluded: int
    '''

    # Read config
    error_threshold_triangulation = config.get('3d-triangulation').get('error_threshold_triangulation')
    min_cameras_for_triangulation = config.get('3d-triangulation').get('min_cameras_for_triangulation')

    # Initialize
    x_files, y_files, likelihood_files = coords_2D_kpt
    n_cams = len(x_files)
    error_min = np.inf
    nb_cams_off = 0  # cameras will be taken-off until the reprojection error is under threshold

    while error_min > error_threshold_triangulation and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(range(n_cams), nb_cams_off)))

        projection_matrices_filt = [projection_matrices] * len(id_cams_off)
        x_files_filt = np.vstack([list(x_files).copy()] * len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()] * len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()] * len(id_cams_off))

        if nb_cams_off > 0:
            for i in range(len(id_cams_off)):
                x_files_filt[i][id_cams_off[i]] = np.nan
                y_files_filt[i][id_cams_off[i]] = np.nan
                likelihood_files_filt[i][id_cams_off[i]] = np.nan
        nb_cams_excluded_filt = [np.count_nonzero(np.nan_to_num(x) == 0) for x in
                                 likelihood_files_filt]  # count nans and zeros

        projection_matrices_filt = [[p[i] for i in range(n_cams) if not np.isnan(x_files_filt[j][i])] for j, p in
                                    enumerate(projection_matrices_filt)]
        x_files_filt = [[xx for ii, xx in enumerate(x) if not np.isnan(xx)] for x in x_files_filt]
        y_files_filt = [[xx for ii, xx in enumerate(x) if not np.isnan(xx)] for x in y_files_filt]
        likelihood_files_filt = [[xx for ii, xx in enumerate(x) if not np.isnan(xx)] for x in likelihood_files_filt]

        # Triangulate 2D points
        Q_filt = [weighted_triangulation(projection_matrices_filt[i], x_files_filt[i], y_files_filt[i],
                                         likelihood_files_filt[i]) for i in range(len(id_cams_off))]

        # Reprojection
        coords_2D_kpt_calc_filt = [reprojection(projection_matrices_filt[i], Q_filt[i]) for i in
                                   range(len(id_cams_off))]
        coords_2D_kpt_calc_filt = np.array(coords_2D_kpt_calc_filt, dtype=object)
        x_calc_filt = coords_2D_kpt_calc_filt[:, 0]
        y_calc_filt = coords_2D_kpt_calc_filt[:, 1]

        # Reprojection error
        error = []
        for config_id in range(len(x_calc_filt)):
            q_file = [(x_files_filt[config_id][i], y_files_filt[config_id][i]) for i in
                      range(len(x_files_filt[config_id]))]
            q_calc = [(x_calc_filt[config_id][i], y_calc_filt[config_id][i]) for i in
                      range(len(x_calc_filt[config_id]))]
            error.append(np.mean([euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))]))

        # Choosing best triangulation (with min reprojection error)
        error_min = min(error)
        best_cams = np.argmin(error)
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]

        Q = Q_filt[best_cams][:-1]

        nb_cams_off += 1

    # If triangulation not successful, error = 0,  and 3D coordinates as missing values
    if error_min > error_threshold_triangulation:
        error_min = np.nan
        # Q = np.array([0.,0.,0.])
        Q = np.array([np.nan, np.nan, np.nan])

    return Q, error_min, nb_cams_excluded


def triangulate_all(config):
    '''
    For each frame
    For each keypoint
    - Triangulate keypoint
    - Reproject it on all cameras
    - Take off cameras until requirements are met
    Interpolate missing values
    Create trc file
    Print recap message
    
     INPUTS: 
    - a calibration file (.toml extension)
    - json files for each camera with only one person of interest
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - a .trc file with 3D coordinates in Y-up system coordinates 
    '''

    # Read config
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    openpose_model = config.get('pose-2d').get('openpose_model')
    pose_folder_name = config.get('project').get('pose_folder_name')
    json_folder_extension = config.get('project').get('pose_json_folder_extension')
    frames_range = config.get('project').get('frames_range')
    likelihood_threshold = config.get('3d-triangulation').get('likelihood_threshold')
    interpolation_kind = config.get('3d-triangulation').get('interpolation')
    pose_dir = os.path.join(project_dir, pose_folder_name)
    poseTracked_folder_name = config.get('project').get('poseTracked_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    poseTracked_dir = os.path.join(project_dir, poseTracked_folder_name)

    # Projection matrix from toml calibration file
    P = computeP(calib_file)

    # Retrieve keypoints from model
    model = eval(openpose_model)
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id != None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id != None]
    keypoints_idx = list(range(len(keypoints_ids)))

    # 2d-pose files selection
    pose_listdirs_names = next(os.walk(pose_dir))[1]
    pose_listdirs_names = natural_sort(pose_listdirs_names)
    json_dirs_names = [k for k in pose_listdirs_names if json_folder_extension in k]

    json_file_sorted_probe = lambda x: int(x.split('.')[0])
    try:
        json_files_names = []
        for js_dir in json_dirs_names:
            before_sort = fnmatch.filter(os.listdir(os.path.join(poseTracked_dir, js_dir)), '*.json')
            before_sort.sort(key=json_file_sorted_probe)
            json_files_names.append(before_sort)
        # print(json_files_names)
        json_tracked_files = [[os.path.join(poseTracked_dir, j_dir, j_file) for j_file in json_files_names[j]] for
                              j, j_dir in enumerate(json_dirs_names)]
    except:
        json_files_names = []
        for js_dir in json_dirs_names:
            before_sort = fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json')
            before_sort.sort(key=json_file_sorted_probe)
            json_files_names.append(before_sort)
        json_tracked_files = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names[j]] for j, j_dir in
                              enumerate(json_dirs_names)]

    # Triangulation
    f_range = [[0, min([len(j) for j in json_files_names])] if frames_range == [] else frames_range][0]

    n_cams = len(json_dirs_names)
    Q_tot, error_tot, nb_cams_excluded_tot = [], [], []
    for f in tqdm(range(*f_range)):
        # Get x,y,likelihood values from files
        json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)]
        x_files, y_files, likelihood_files = extract_files_frame_f(json_tracked_files_f, keypoints_ids)

        # Replace likelihood by 0 if under likelihood_threshold
        with np.errstate(invalid='ignore'):
            likelihood_files[likelihood_files < likelihood_threshold] = 0.

        Q, error, nb_cams_excluded = [], [], []
        for keypoint_idx in keypoints_idx:
            # Triangulate cameras with min reprojection error
            coords_2D_kpt = (x_files[:, keypoint_idx], y_files[:, keypoint_idx], likelihood_files[:, keypoint_idx])
            Q_kpt, error_kpt, nb_cams_excluded_kpt = triangulation_from_best_cameras(config, coords_2D_kpt, P)

            Q.append(Q_kpt)
            error.append(error_kpt)
            nb_cams_excluded.append(nb_cams_excluded_kpt)

        # Add triangulated points, errors and excluded cameras to pandas dataframes
        Q_tot.append(np.concatenate(Q))
        error_tot.append(error)
        nb_cams_excluded_tot.append(nb_cams_excluded)

    Q_tot = pd.DataFrame(Q_tot)
    error_tot = pd.DataFrame(error_tot)
    nb_cams_excluded_tot = pd.DataFrame(nb_cams_excluded_tot)

    error_tot['mean'] = error_tot.mean(axis=1)
    nb_cams_excluded_tot['mean'] = nb_cams_excluded_tot.mean(axis=1)

    # Interpolate missing values
    Q_tot = Q_tot.apply(interpolate_zeros, axis=0, args=[interpolation_kind])

    # Create TRC file
    trc_path = make_trc(config, Q_tot, keypoints_names, f_range)

    # Recap message
    recap_triangulate(config, error_tot, nb_cams_excluded_tot, keypoints_names, trc_path)
