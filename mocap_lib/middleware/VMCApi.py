# -- coding: utf-8 --
# @Time : 2022/2/14
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from osc4py3.as_eventloop import *  # osc module
from osc4py3 import oscbuildparse

# import time

LEFT_MPII_HAND_LABELS = [
    'LEFT_WRIST',  # 0
    'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP',
    'LEFT_INDEX_FINGER_MCP', 'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP',
    'LEFT_MIDDLE_FINGER_MCP', 'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP',
    'LEFT_RING_FINGER_MCP', 'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP',
    'LEFT_PINKY_MCP', 'LEFT_PINKY_PIP', 'LEFT_PINKY_DIP', 'LEFT_PINKY_TIP',
]

RIGHT_MPII_HAND_LABELS = [
    'RIGHT_WRIST',  # 0
    'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP',
    'RIGHT_INDEX_FINGER_MCP', 'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP',
    'RIGHT_MIDDLE_FINGER_MCP', 'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP',
    'RIGHT_RING_FINGER_MCP', 'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP',
    'RIGHT_PINKY_MCP', 'RIGHT_PINKY_PIP', 'RIGHT_PINKY_DIP', 'RIGHT_PINKY_TIP',
]

LEFT_UNI_HAND_LABELS = [
    'LEFT_WRIST',  # 0
    'LEFT_THUMB_CMC', 'LeftThumbProximal', 'LeftThumbIntermediate', 'LeftThumbDistal',
    'LEFT_INDEX_FINGER_MCP', 'LeftIndexProximal', 'LeftIndexIntermediate', 'LeftIndexDistal',
    'LEFT_MIDDLE_FINGER_MCP', 'LeftMiddleProximal', 'LeftMiddleIntermediate', 'LeftMiddleDistal',
    'LEFT_RING_FINGER_MCP', 'LeftRingProximal', 'LeftRingIntermediate', 'LeftRingDistal',
    'LEFT_PINKY_MCP', 'LeftLittleProximal', 'LeftLittleIntermediate', 'LeftLittleDistal',
]

RIGHT_UNI_HAND_LABELS = [
    'Right_WRIST',  # 0
    'Right_THUMB_CMC', 'RightThumbProximal', 'RightThumbIntermediate', 'RightThumbDistal',
    'Right_INDEX_FINGER_MCP', 'RightIndexProximal', 'RightIndexIntermediate', 'RightIndexDistal',
    'Right_MIDDLE_FINGER_MCP', 'RightMiddleProximal', 'RightMiddleIntermediate', 'RightMiddleDistal',
    'Right_RING_FINGER_MCP', 'RightRingProximal', 'RightRingIntermediate', 'RightRingDistal',
    'Right_PINKY_MCP', 'RightLittleProximal', 'RightLittleIntermediate', 'RightLittleDistal',
]


class VMCApi:
    def __init__(self, ip_address, ip_port):
        # ip = '192.168.4.13'  # ip address
        # port = 39539  # port number

        osc_startup()  # starts osc protocol
        osc_udp_client(ip_address, ip_port, "VroidPoser")  # initializes osc client

    def sendosc(self, bone, x, y, z, w):  # condensed OSC message function

        msg = oscbuildparse.OSCMessage("/VMC/Ext/Bone/Pos", None,
                                       [bone, float(0), float(0), float(0), float(x),
                                        float(y), float(z), float(w)])
        # print(msg)
        osc_send(msg, "VroidPoser")
        osc_process()
