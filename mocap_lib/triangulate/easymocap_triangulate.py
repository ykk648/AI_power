# -- coding: utf-8 --
# @Time : 2022/7/18
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np


def projectN3(kpts3d, Pall):
    # kpts3d: (N, 3)
    nViews = len(Pall)
    kp3d = np.hstack((kpts3d[:, :3], np.ones((kpts3d.shape[0], 1))))
    kp2ds = []
    for nv in range(nViews):
        kp2d = Pall[nv] @ kp3d.T
        kp2d[:2, :] /= kp2d[2:, :]
        kp2ds.append(kp2d.T[None, :, :])
    kp2ds = np.vstack(kp2ds)
    if kpts3d.shape[-1] == 4:
        kp2ds[..., -1] = kp2ds[..., -1] * (kpts3d[None, :, -1] > 0.)
    return kp2ds


def batch_triangulate(keypoints_, Pall, keypoints_pre=None, lamb=1e3):
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1] > 0).sum(axis=0)
    valid_joint = np.where(v > 1)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0) / v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    if keypoints_pre is not None:
        # keypoints_pre: (nJoints, 4)
        B = np.eye(4)[None, :, :].repeat(A.shape[0], axis=0)
        B[:, :3, 3] = -keypoints_pre[valid_joint, :3]
        confpre = lamb * keypoints_pre[valid_joint, 3]
        # 1, 0, 0, -x0
        # 0, 1, 0, -y0
        # 0, 0, 1, -z0
        # 0, 0, 0,   0
        B[:, 3, 3] = 0
        B = B * confpre[:, None, None]
        A = np.hstack((A, B))
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d
    return result


def simple_recon_person(keypoints_use, Puse):
    out = batch_triangulate(keypoints_use, Puse)
    # compute reprojection error
    kpts_repro = projectN3(out, Puse)
    # square_diff = (keypoints_use[:, :, :2] - kpts_repro[:, :, :2]) ** 2
    conf = np.repeat(out[None, :, -1:], len(Puse), 0)
    kpts_repro = np.concatenate((kpts_repro, conf), axis=2)
    return out, kpts_repro


def check_repro_error(keypoints3d, kpts_repro, keypoints2d, P, MAX_REPRO_ERROR):
    # square_diff = (keypoints2d[:, :, :2] - kpts_repro[:, :, :2]) ** 2
    # conf = keypoints3d[None, :, -1:]
    conf = (keypoints3d[None, :, -1:] > 0) * (keypoints2d[:, :, -1:] > 0)
    dist = np.sqrt((((kpts_repro[..., :2] - keypoints2d[..., :2]) * conf) ** 2).sum(axis=-1))
    # print(dist.mean())
    vv, jj = np.where(dist > MAX_REPRO_ERROR)
    if vv.shape[0] > 0:
        keypoints2d[vv, jj, -1] = 0.
        keypoints3d, kpts_repro = simple_recon_person(keypoints2d, P)
    return keypoints3d, kpts_repro


class EasyMocapTriangulate:
    def __init__(self, pkl_path_, pkl_mode_='multical', max_repro_error=70):
        self.max_repro_error = max_repro_error

        if pkl_mode_ == 'multical':
            cvc = CVCamera(pkl_path_)
            self.pall = cvc.multi_cam_stack(cvc.pall())
        elif pkl_mode_ == 'anipose':
            cvc = CVCamera(None)
            c_group = CVFile(pkl_path_).data
            self.pall = cvc.multi_cam_stack(cvc.pall_from_cgroup(c_group))

        # self.pall = cvc.multi_cam_stack(cvc.pall_rotate())

    def triangulate(self, multi_view_kps):
        """

        Args:
            multi_view_kps: N_view * N_kps * 3

        Returns:
            kps_3d: N_kps * 4
            kps_repro: N_view * N_kps * 4
        """
        kps_3d, kps_repro = simple_recon_person(multi_view_kps, self.pall)
        kps_3d, kps_repro = check_repro_error(kps_3d, kps_repro, multi_view_kps, P=self.pall,
                                              MAX_REPRO_ERROR=self.max_repro_error)
        return kps_3d, kps_repro


if __name__ == '__main__':
    import cv2
    from cv2box import CVFile, CVVideoLoader, CVImage, CVCamera

    # using 2d kps to bundle adjust cam params
    cam_pkl_path = ''
    kp_2d_pkl_list = []
    keypoints_2d = []
    for i in range(len(kp_2d_pkl_list)):
        kp_2d_this_cam = CVFile(kp_2d_pkl_list[i]).data
        keypoints_2d.append(kp_2d_this_cam)
    keypoints_2d = np.array(keypoints_2d)
    camera_group = CVCamera(cam_pkl_path).load_camera_group().bundle_adjust_iter(keypoints_2d)
    CVFile('cgroup.pkl').pickle_write(camera_group)

    # triangulate
    emt = EasyMocapTriangulate(cam_pkl_path)
    for i in range(901):
        keypoints_2d_ = keypoints_2d[:, i, :, :].reshape((4, 67, 3))
        kps_3d, kps_repro = emt.triangulate(keypoints_2d_)
        CVFile('./{}.pkl'.format(i)).pickle_write(kps_3d)

    # # anipose交叉验证，需要原始pall代替rotate pall
    # from mocap_lib.triangulate.anipose_triangulate import AniposeTriangulate
    # at = AniposeTriangulate(cam_pkl_path)
    # n_view_kps_2d = at.project(kps_3d[:, :3])
    #
    # with CVVideoLoader('') as cvvl:
    #     for i in range(len(cvvl)):
    #         _, frame = cvvl.get()
    #         for j in range(kps_3d.shape[0]):
    #             try:
    #                 # print((int(kps_repro[0][j][0]), int(kps_repro[0][j][1])))
    #                 cv2.circle(frame, (int(kps_repro[1][j][0]), int(kps_repro[1][j][1])), 4, color=(255, 0, 0))
    #             except:
    #                 pass
    #         CVImage(frame).show(0)
