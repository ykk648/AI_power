# -- coding: utf-8 --
# @Time : 2022/7/19
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from mocap_lib.skeleton_transfer.bone_links import BONE_CONFIG
from cv2box.utils import get_path_by_ext


def get_skeleton_form_kintree(kintree):
    length = 0
    for i, j in kintree:
        if length < i:
            length = i
        if length < j:
            length = j
    length = length + 1
    parents = np.ones(length, dtype=np.uint8) * -1
    for i, j in kintree:
        parents[i] = j
    return parents


def render_animation(kintree, poses, output=None, fps=30, bitrate=0, azim=-45, elev=20.,
                     limit=-1, size=6, input_video_skip=0,
                     input_file_names=None, tracking=False):
    """
    Args:
        kintree:
        poses: N_frame * N_kp_num * 3
        output:
        fps:
        bitrate:
        azim:
        elev:
        limit:
        size:
        input_video_skip:
        input_file_names:
        tracking:
    Returns:
    """
    if output is not None:
        plt.ioff()
    else:
        plt.ion()
    fig = plt.figure(figsize=(size * (len(poses)), size))

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 2
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for index, (title, data) in enumerate(poses.items()):
        # center_x = np.sum(data[:, 8, 0]) / np.sum(data[:, 8, 2] != 0)
        # center_z = np.sum(data[:, 8, 1]) / np.sum(data[:, 8, 2] != 0)
        center_x = 0.25
        center_z = 0.35
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-radius, radius])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius, radius])
        if not tracking:
            ax.set_xlim3d([-radius + center_x, radius + center_x])
            ax.set_ylim3d([-radius + center_z, radius + center_z])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 2]])
    poses = list(poses.values())
    if limit == -1:
        limit = poses[0].shape[0]
    else:
        limit = np.minimum(limit, poses[0].shape[0])

    # Decode video
    initialized = False
    # image = None
    # lines = []
    # points = None

    parents = get_skeleton_form_kintree(kintree)
    joints_right = []

    def update_video(i):
        nonlocal initialized  # image, lines, points
        if tracking:
            for n, ax in enumerate(ax_3d):
                ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
                ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
        if not initialized:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                for n, ax in enumerate(ax_3d):
                    if n == 0:
                        # GT
                        ls = (0, ())
                        col = 'red' if j in joints_right else 'black'
                    else:
                        # prediction
                        ls = (0, (3, 1, 1, 1, 1, 1))
                        col = 'blue' if j in joints_right else 'green'
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 2], pos[j_parent, 2]],
                                               [-pos[j, 1], -pos[j_parent, 1]], zdir='z', c=col, linestyle=ls))
            initialized = True
        else:
            idx = 0
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if pos[j, 0] != 0 and pos[j_parent, 1] != 0 and pos[j_parent, 2] != 0:
                        lines_3d[n][idx][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        lines_3d[n][idx][0].set_ydata(np.array([pos[j, 2], pos[j_parent, 2]]))
                        lines_3d[n][idx][0].set_3d_properties(np.array([-pos[j, 1], -pos[j_parent, 1]]), zdir='z')
                    else:
                        lines_3d[n][idx][0].set_xdata(np.array([0, 0]))
                        lines_3d[n][idx][0].set_ydata(np.array([0, 0]))
                        lines_3d[n][idx][0].set_3d_properties(np.array([0, 0]), zdir='z')
                idx += 1
        print('{}/{}      '.format(i, limit), end='\r')
        if output is None:
            plt.show()

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
    if output is not None:
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()


if __name__ == '__main__':
    from cv2box import CVFile
    input_path = '/workspace/codes/hrnet48_realtime/results/3d_kp'
    output_path = './cache/3d_vis/result.mp4'

    kp3ds = []
    Z = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape((3, 3))

    # for 2d kp in a dir
    for file in get_path_by_ext(input_path, ['.pkl']):
        # 坐标变换
        kp3ds.append(np.load(file, allow_pickle=True)[ :, :3])
        # kp3ds.append(np.load(file, allow_pickle=True)[None][:, :, :3] @ Z.T)

    # # for 3d kp in a pkl
    # kp3ds = np.array(CVFile(input_path).data)
    # kp3ds = optim_3d @ Z.T

    kp3ds = np.array(kp3ds)
    print(kp3ds.shape)
    points_dict = {'1': kp3ds}
    # openpose_body25 openpose_bodyhand67
    render_animation(BONE_CONFIG['openpose_bodyhand67']['kintree'], points_dict, output_path, limit=-1)
