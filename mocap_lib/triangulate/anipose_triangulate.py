# -- coding: utf-8 --
# @Time : 2022/7/15
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVFile, CVCamera
import numpy as np

from mocap_lib.skeleton_transfer.bone_links import BONE_CONFIG


class AniposeTriangulate:
    def __init__(self, pkl_path_, pkl_mode_='anipose'):
        if pkl_mode_ == 'multical':
            self.c_group = CVCamera(pkl_path_).load_camera_group()
        elif pkl_mode_ == 'anipose':
            self.c_group = CVFile(pkl_path_).data

    def triangulate(self, multi_view_kps):
        """

        Args:
            multi_view_kps: N_view * N_kps * 3

        Returns:
            kps_3d: N_kps * 3

        """
        # multi_view_kps = multi_view_kps[:, :, 0:2]

        # # default total 60fps
        # kps_3d = self.c_group.triangulate(multi_view_kps, undistort=True, progress=False)

        # # ransac slow total 10fps
        # kps_3d = self.c_group.triangulate_ransac(multi_view_kps, progress=False)[0]

        # # offline  shape CxNxJx2
        nframes = multi_view_kps.shape[0]
        kps_3d = self.c_group.triangulate_optim(multi_view_kps,
                                                constraints=BONE_CONFIG['openpose_bodyhand67']['kintree'],
                                                verbose=True, init_progress=True)

        return kps_3d

    def project(self, kps_3d_):
        """

        Args:
            kps_3d_: N_kps * 3

        Returns:
            kps_2d: N_view * N_kps * 2

        """
        kps_2d = self.c_group.project(kps_3d_)
        return kps_2d


if __name__ == '__main__':
    from cv2box.utils import get_path_by_ext

    # triangulate
    at = AniposeTriangulate('cgroup.pkl')
    kp2ds = []
    for file in get_path_by_ext('', ['.pkl']):
        # 坐标变换
        kp2ds.append(np.load(file, allow_pickle=True))
        # kp3ds.append(np.load(file, allow_pickle=True)[None])
    kp2ds = np.array(kp2ds).transpose((1, 0, 2, 3))[:, 600:900, :, :2]
    results = at.triangulate(kp2ds)
    CVFile('3dkp_optim.pkl').pickle_write(results)

