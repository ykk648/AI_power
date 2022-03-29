# -- coding: utf-8 --
# @Time : 2022/2/14
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from model_lib.onnx_model import ONNXModel
import numpy as np
from estimation_3d.hand_mesh.minimal_hands.kinematics import xyz_to_delta, MPIIHandJoints, mano_to_mpii
from cv2box import CVFile

IK_UNIT_LENGTH = 0.09473151311686484
mano_ref_xyz = CVFile('pretrain_models/digital_human/minimal_hands/hand_mesh_model.pkl').data['joints']
# convert the kinematic definition to MPII style, and normalize it
mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / IK_UNIT_LENGTH
mpii_ref_xyz -= mpii_ref_xyz[9:10]
# get bone orientations in the reference pose
mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
mpii_ref_delta = mpii_ref_delta * mpii_ref_length


class IKModel:
    def __init__(self):
        self.ik_model = ONNXModel('pretrain_models/digital_human/minimal_hands/iknet/iknet.onnx')

    def forward_np(self, hand_np):
        # xyz = np.array(hand_np)
        delta, length = xyz_to_delta(hand_np, MPIIHandJoints)
        delta *= length
        pack = np.concatenate(
            [hand_np, delta, mpii_ref_xyz, mpii_ref_delta], 0
        )
        return self.forward(pack)

    def forward(self, pack):
        pack = np.expand_dims(pack, 0)
        theta = self.ik_model.forward(pack.astype(np.float32))[0]
        # theta_mano = mpii_to_mano(theta)
        if len(theta.shape) == 3:
            theta = theta[0]
        return theta
