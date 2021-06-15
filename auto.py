#!/usr/bin/env python3

import asyncio
import cv2
import gi
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import socket
import time
import websockets

from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)

gi.require_version('Gst', '1.0')
from gi.repository import Gst


# {{{ class Video():
class Video():
    """BlueRov video capture class constructor
    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self, port=5600):
        """Summary
        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self._frame = None

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]
        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]

        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array
        Args:
            sample (TYPE): Description
        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray((caps.get_structure(0).get_value('height'),
                            caps.get_structure(0).get_value('width'), 3),
                           buffer=buf.extract_dup(0, buf.get_size()),
                           dtype=np.uint8)
        return array

    async def frame(self):
        """ Get Frame
        Returns:
            iterable: bool and image frame, cap.read() output
        """
        return self._frame

    def frame_available(self):
        """Check if frame is available
        Returns:
            bool: true if frame is available
        """
        return type(self._frame) != type(None)

    def run(self):
        """ Get frame to update _frame
        """

        self.start_gst([
            self.video_source, self.video_codec, self.video_decode,
            self.video_sink_conf
        ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame

        return Gst.FlowReturn.OK


# }}}


# {{{ CV
def wrapping(image):
    (h, w) = (image.shape[0], image.shape[1])  # 360:640
    Orimid = w / 2

    a = [0.3 * w, 0.7 * h]
    b = [0.7 * w, 0.7 * h]
    c = [0.15 * w, 0.95 * h]
    d = [0.85 * w, 0.95 * h]
    enddist = abs(a[0] - b[0])
    oridist = abs(c[0] - d[0])
    cmdist = abs(c[0] - Orimid)

    # warp된 이미지에서 찾을 수 있는 원본이미지의 중간
    warpmid = (int(w * cmdist / oridist), h)
    distlist = (enddist, a, b, h, w)

    source = np.float32([a, b, c, d])  #input point  #좌상, 우상, 좌하, 우하
    destination = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  #output Point
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    minv = cv2.getPerspectiveTransform(
        destination, source)  #minv = warpping된 이미지를 원금감을 주기 위한 반다의 matrix값을 저장
    _image = cv2.warpPerspective(image, transform_matrix, (w, h))

    return _image, minv, warpmid, distlist


def color_filter(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.array([0, 40, 0])
    upper = np.array([120, 100, 110])

    mask = cv2.inRange(hls, lower, upper)
    masked = cv2.bitwise_and(image, image, mask=mask)

    return masked


def roi(image):
    x = int(image.shape[1])
    y = int(image.shape[0])

    # 한 붓 그리기
    _shape = np.array([[int(0.01 * x), int(y)], [int(0.01 * x),
                                                 int(0.1 * y)],
                       [int(0.4 * x), int(0.1 * y)], [int(0.4 * x),
                                                      int(y)],
                       [int(0.55 * x), int(y)], [int(0.55 * x),
                                                 int(0.1 * y)],
                       [int(0.90 * x), int(0.1 * y)], [int(0.90 * x),
                                                       int(y)],
                       [int(0.2 * x), int(y)]])
    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, np.int32([_shape]), ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def plothistogram(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    midpoint = np.int_(histogram.shape[0] / 2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint

    return leftbase, rightbase


def slide_window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nwindows = 4
    window_height = np.int_(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
    margin = 100
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    for w in range(nwindows):
        win_y_low = binary_warped.shape[0] - (w +
                                              1) * window_height  # window 윗부분
        win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
        win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
        win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
        win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
        win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), color, thickness)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), color, thickness)
        good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                     (nonzero_x >= win_xleft_low) &
                     (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                      (nonzero_x >= win_xright_low) &
                      (nonzero_x < win_xright_high)).nonzero()[0]
        left_lane.append(good_left)
        right_lane.append(good_right)
        # cv2.imshow("oo", out_img)

        if len(good_left) > minpix:
            left_current = int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = int(np.mean(nonzero_x[good_right]))

    left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
    right_lane = np.concatenate(right_lane)

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    if len(lefty) and len(leftx) and len(righty) and len(rightx):  #라인인식 하면 실행
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1,
                            binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[
            2]

        ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
        rtx = np.trunc(right_fitx)

        out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
        out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

        ret = {'left_fitx': ltx, 'right_fitx': rtx, 'ploty': ploty}

        return ret
    return 90  #라인인식 못 하면 90도를 리턴


def draw_lane_lines(original_image, warped_image, Minv, draw_info, warpmid,
                    distlist):
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

    cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))

    warpend = (np.int_([pts_mean[0][-1][0]]), np.int_([pts_mean[0][-1][1]])
              )  #가야할 곳의 끝점
    endpoint = (int(distlist[1][0] + distlist[0] * warpend[0] / distlist[4]),
                int(distlist[1][1]))  #원본이미지의 끝점
    widthend = (endpoint[0], distlist[3])  #끝점에서 직각으로 내려오는 점

    cv2.line(original_image, endpoint, endpoint, (0, 255, 255), 10,
             4)  # 두 점을 잇는 선 표시

    frame_size = (original_image.shape[1], original_image.shape[0])
    if frame_size[0] / 2 == endpoint[0]:
        res = 90
    elif frame_size[0] / 2 > endpoint[0]:
        res = math.atan((frame_size[1] - endpoint[1]) /
                        ((frame_size[0] / 2) - endpoint[0])) / math.pi * 180
    else:
        res = (1 - math.atan(
            (frame_size[1] - endpoint[1]) /
            (endpoint[0] - (frame_size[0] / 2))) / math.pi) * 180

    #각도 예외 처리 30도 이상의 각도 변화 없앰
    if res < 60 or res > 120:
        res = 90
    # print(endpoint)
    # print(res)  #각도

    newwarp = cv2.warpPerspective(
        color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    cv2.imshow("asdad", color_warp)
    result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)

    return pts_mean, result, res


# }}}


class Drone:
    t = float("inf")

    def __init__(self):
        self.yaw = 90
        self.deg = 0

    async def run(self):
        drone = System()
        await drone.connect(system_address="udp://:14540")

        print("-- Arming")
        try:
            await drone.action.arm()
        except Exception as e:
            print(e)

        print("-- Setting initial setpoint")
        N_loc = 0
        E_loc = 0
        D_loc = -1.2
        await drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 0.0, 0.0, self.yaw))

        print("-- Starting offboard")
        try:
            await drone.offboard.start()
        except OffboardError as error:
            print(
                f"Starting offboard mode failed with error code: {error._result.result}"
            )
            print("-- Disarming")
            await drone.action.disarm()
            return

        interval = 0.2
        velocity = 2  # m/s
        print(f"-- {-D_loc}m 상승")
        await drone.offboard.set_position_ned(
            PositionNedYaw(N_loc, E_loc, D_loc, self.yaw))
        await asyncio.sleep(5)

        start_time = time.monotonic()
        while time.monotonic() < start_time + Drone.t:
            self.yaw += self.deg
            N_loc += interval * velocity * math.cos(math.pi * self.yaw / 180)
            E_loc += interval * velocity * math.sin(math.pi * self.yaw / 180)
            await drone.offboard.set_position_ned(
                PositionNedYaw(N_loc, E_loc, D_loc, self.yaw))
            await asyncio.sleep(interval)

        print("-- Stopping offboard")
        try:
            await drone.offboard.stop()
        except OffboardError as error:
            print(
                f"Stopping offboard mode failed with error code: {error._result.result}"
            )

        await asyncio.sleep(5)

        await drone.action.land()

    async def cv(self):
        video = Video()

        while True:
            if not video.frame_available():
                continue

            img = await video.frame()
            cv2.imshow('video', img)

            wrapped_img, minverse, warpmid, distlist = wrapping(img)
            cv2.imshow('wrapped', wrapped_img)

            ## 조감도 필터링
            w_f_img = color_filter(wrapped_img)
            cv2.imshow('w_f_img', w_f_img)

            w_f_r_img = roi(w_f_img)
            cv2.imshow('w_f_r_img', w_f_r_img)

            ## 조감도 선 따기 wrapped img threshold
            _gray = cv2.cvtColor(w_f_r_img, cv2.COLOR_BGR2GRAY)
            # ret, thresh = cv2.threshold(_gray, 200, 255, cv2.THRESH_BINARY)
            ret, thresh = cv2.threshold(_gray, 10, 255, cv2.THRESH_BINARY)
            cv2.imshow('threshold', thresh)

            blurred = cv2.GaussianBlur(thresh, (11, 11), 0)
            thresh = blurred
            ## 선 분포도 조사 histogram
            leftbase, rightbase = plothistogram(thresh)

            ## histogram 기반 window roi 영역
            draw_info = slide_window_search(thresh, leftbase, rightbase)

            if draw_info != 90:
                ## 원본 이미지에 라인 넣기
                meanPts, result, res = draw_lane_lines(img, thresh, minverse,
                                                       draw_info, warpmid,
                                                       distlist)
                cv2.imshow("result", result)
            else:
                res = 90
                # print(draw_info)

            self.deg = min(max(res - 90, -5), 5)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            await asyncio.sleep(0.03)

        cv2.destroyAllWindows()

    @staticmethod
    async def accept(websocket, path):
        while True:
            data_rcv = await websocket.recv()
            json_rcv = json.loads(data_rcv)
            print("received data = " + data_rcv)
            try:
                t = int(json_rcv["time"])
                Drone.t = t
            except:
                pass
            await websocket.send("websocket_svr send data = " + data_rcv)
            await asyncio.sleep(0.0)

    # @staticmethod
    # async def runserver():

    #     async def handle_client(reader, writer):
    #         request = None
    #         while request != 'quit':
    #             request = (await reader.read(255)).decode('utf8')
    #             response = str(eval(request)) + '\n'
    #             writer.write(response.encode('utf8'))
    #             try:
    #                 t = int(request)
    #                 Drone.t = t
    #             except:
    #                 pass
    #             await writer.drain()
    #             await asyncio.sleep(0.0)
    #         writer.close()

    #     server = await asyncio.start_server(handle_client, '0.0.0.0', 8808)
    #     async with server:
    #         await server.serve_forever()
    #         await asyncio.sleep(0.0)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    d = Drone()
    websoc_svr = websockets.serve(d.accept, "127.0.0.1", 8282)
    # loop.run_until_complete(websoc_svr)
    loop.create_task(websoc_svr)
    loop.create_task(d.run())
    loop.create_task(d.cv())
    loop.run_forever()
