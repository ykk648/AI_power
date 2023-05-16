# -- coding: utf-8 --
# @Time : 2022/5/19
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power


# import required libraries
from .vidgear.camgear import CamGear
import cv2
import time
import datetime
# from cv2box import CVImage, MyFpsCounter, MyTimer


class ReconnectingCamGear:
    def __init__(self, reset_attempts=50, reset_delay=5):
        self.reset_attempts = reset_attempts
        self.reset_delay = reset_delay

        self.input_option_dict = {
            # 'CAP_PROP_FRAME_WIDTH': 1280,
            # 'CAP_PROP_FRAME_HEIGHT': 720,
            # 'CAP_PROP_FPS': 30,
            # 'CAP_PROP_FOURCC': cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            # 'CAP_PROP_FOURCC': cv2.VideoWriter_fourcc('M', 'P', '4', '2'),
            # 'CAP_PROP_FOURCC': cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
            # 'CAP_PROP_FOURCC': cv2.VideoWriter_fourcc('H', '2', '6', '4'),
            'THREADED_QUEUE_MODE': True,  # default
        }
        self.rtmpA = 'rtmp://localhost:1935/gopro0/movie'
        self.rtmpB = 'rtmp://localhost:1935/gopro1/movie'
        self.rtmpC = 'rtmp://localhost:1935/gopro2/movie'
        self.rtmpD = 'rtmp://localhost:1935/gopro3/movie'

        # start_time = time.time()
        self.sourceA = CamGear(source=self.rtmpA, logging=True, time_delay=0, **self.input_option_dict).start()
        a_time = self.sourceA.get_first_grab_time()
        self.sourceB = CamGear(source=self.rtmpB, logging=True, time_delay=0, **self.input_option_dict).start()
        b_time = self.sourceB.get_first_grab_time()
        self.sourceC = CamGear(source=self.rtmpC, logging=True, time_delay=0, **self.input_option_dict).start()
        c_time = self.sourceC.get_first_grab_time()
        self.sourceD = CamGear(source=self.rtmpD, logging=True, time_delay=0, **self.input_option_dict).start()
        d_time = self.sourceD.get_first_grab_time()

        self.pass_frame_number_a = round((d_time - a_time) / (1 / 30))  # int round
        self.pass_frame_number_b = round((d_time - b_time) / (1 / 30))
        self.pass_frame_number_c = round((d_time - c_time) / (1 / 30))

        print('frame offset: ', self.pass_frame_number_a, self.pass_frame_number_b, self.pass_frame_number_c)

        self.running = True
        self.first = True
        # self.framerateA = self.sourceA.framerate

    def read(self):
        if self.sourceA is None or self.sourceB is None:
            return None
        if self.running and self.reset_attempts > 0:
            if self.first:
                # offset
                while self.pass_frame_number_a > 0:
                    _ = self.sourceA.read()
                    self.pass_frame_number_a -= 1
                while self.pass_frame_number_b > 0:
                    _ = self.sourceB.read()
                    self.pass_frame_number_b -= 1
                while self.pass_frame_number_c > 0:
                    _ = self.sourceC.read()
                    self.pass_frame_number_c -= 1
                self.first = False

            frameA = self.sourceA.read()
            frameB = self.sourceB.read()
            frameC = self.sourceC.read()
            frameD = self.sourceD.read()

            if frameA is None or frameB is None:
                self.sourceA.stop()
                self.sourceB.stop()
                self.sourceC.stop()
                self.sourceD.stop()
                self.reset_attempts -= 1
                print(
                    "Re-connection Attempt-{} occured at time:{}".format(
                        str(self.reset_attempts),
                        datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"),
                    )
                )
                time.sleep(self.reset_delay)
                self.sourceA = CamGear(source=self.rtmpA).start()
                self.sourceB = CamGear(source=self.rtmpB).start()
                self.sourceC = CamGear(source=self.rtmpC).start()
                self.sourceD = CamGear(source=self.rtmpD).start()
                # return previous frame
                return self.frameA, self.frameB, self.frameC, self.frameD
            else:
                self.frameA = frameA
                self.frameB = frameB
                self.frameC = frameC
                self.frameD = frameD
                return frameA, frameB, frameC, frameD
        else:
            return None

    def stop(self):
        self.running = False
        self.reset_attempts = 0
        self.frame = None
        if not self.sourceA is None and not self.sourceB is None and not self.sourceC is None and not self.sourceD is None:
            self.sourceA.stop()
            self.sourceB.stop()
            self.sourceC.stop()
            self.sourceD.stop()


if __name__ == '__main__':

    outVideo1 = cv2.VideoWriter('stream1.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1920, 1080))
    outVideo2 = cv2.VideoWriter('stream2.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1920, 1080))
    outVideo3 = cv2.VideoWriter('stream3.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1920, 1080))
    outVideo4 = cv2.VideoWriter('stream4.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (1920, 1080))
    count = 0

    time.sleep(1)
    print('start recording')

    stream_combine = ReconnectingCamGear(reset_attempts=20, reset_delay=5, )

    # infinite loop
    while True:
        frameA_, frameB_, frameC_, frameD_ = stream_combine.read()

        if frameA_ is None or frameB_ is None or frameC_ is None or frameD_ is None:
            break
        outVideo1.write(frameA_)
        outVideo2.write(frameB_)
        outVideo3.write(frameC_)
        outVideo4.write(frameD_)
        count += 1
        if count > 300:
            break
        # CVImage(frameA).show(wait_time=1)

    cv2.destroyAllWindows()
    stream_combine.stop()
    outVideo1.release()
    outVideo2.release()
    outVideo3.release()
    outVideo4.release()
