# live_streaming.py

from flask import Flask, render_template, Response, request, jsonify
import json
import object_detector
import camera
import cv2
import datetime
import time
import logging
import server_conf
import face_recognition
import numpy as np
import os

class Live():
    def __init__(self):
        self.host = '0.0.0.0'
        self.port = server_conf.live_port
        self.debug = True
        self.current_time = 30 # second
        self.current_buffer = []
        self.app = Flask(__name__)
        self.init_flask(self.app)
        self.logger = self.get_logger("live_server")
        self.retry_cnt_max = 5

    def get_logger(self, logger_name, logging_level=logging.DEBUG):
        """ 로거를 만들어줌 """
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging_level)
        file_handler = logging.FileHandler(f'./logs/{logger_name}.log')
        formatter = logging.Formatter("[%(levelname)s] '%(filename)s' %(asctime)s : %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def init_flask(self, app):

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/video_feed')
        def video_feed():
            return Response(gen(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        def gen():
            camera = VideoCamera(server_conf.camera_source)
            detector = object_detector.ObjectDetector('ssd_mobilenet_v1_coco_2017_11_17')
            face_recog_m = FaceRecog()
            self.logger.debug("start detector")
            #detector = ObjectDetector('mask_rcnn_inception_v2_coco_2018_01_28')
            #detector = ObjectDetector('pet', label_file='data/pet_label_map.pbtxt')

            retry_cnt = 0
            while True:
                try:
                    frame = camera.get_frame()
                    frame, face_result_list = face_recog_m.get_frame_live(frame)
                    #frame = cam.get_frame()
                    frame, obj_detect_dict = detector.detect_objects_live(frame)
                    self.buffer_handle(face_result_list, obj_detect_dict)

                    ret, jpg = cv2.imencode('.jpg', frame)
                    jpg_bytes = jpg.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')
                except Exception as err:
                    self.logger.error(str(err))
                    try:
                        camera = VideoCamera(server_conf.camera_source)
                    except:
                        retry_cnt += 1
                        if retry_cnt > self.retry_cnt_max:
                            break

        @app.route("/health", methods=["GET"])
        def health_check():
            method = request.method
            self.logger.debug("health check")
            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": True,
                      "message": "live streaming server health check success"
                    }
                }
            return json.dumps(rtn)

        @app.route("/show", methods=["GET"])
        def show():
            method = request.method
            self.logger.debug("show")
            obj_list = self.show_current_all()
            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": True,
                      "message": str(obj_list)
                    }
                }
            return json.dumps(rtn)

    def show_current_all(self):
        rtn_set = set([])
        for item in self.current_buffer:
            rtn_set.add(item[1])
        return list(rtn_set)


    def buffer_handle(self, face_result_list, obj_detect_dict):
        now_date = datetime.datetime.now()
        now_str = now_date.strftime("%Y%m%d%H%M%S")

        for face_result in face_result_list:
            face_corr = face_result[0]
            face_name = face_result[1]
            self.current_buffer.append((now_date, face_name))
            if self.check_current_max(now_date):
                del self.current_buffer[0]
        for item in obj_detect_dict:
            try:
                obj_corr = item
                class_str = obj_detect_dict[item][0]
                class_list = class_str.split()
                obj_class = class_list[0][:-1]
                obj_score = int(class_list[1][:-1])
                self.current_buffer.append((now_date, obj_class))
                if self.check_current_max(now_date):
                    del self.current_buffer[0]
            except:
                pass

    def check_current_max(self, now_date):
        if len(self.current_buffer) > 0:
            last_date_time = self.current_buffer[0][0]
            if now_date - last_date_time >= datetime.timedelta(minutes=self.current_time):
                return True
        return False

    def run(self):
        self.app.run(host=self.host, port=self.port, debug=self.debug)

class VideoCamera(object):
    def __init__(self, camera_source=0):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        self.video = cv2.VideoCapture(camera_source)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # Grab a single frame of video
        ret, frame = self.video.read()
        return frame

    def get_size(self):
        return (self.video.get(cv2.CAP_PROP_FRAME_WIDTH),
                self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.

        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def get_frame(self, frame):

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_frame_live(self, frame):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        result_list = []
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            result_list.append(((top, right, bottom, left), name))
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame, result_list

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

if __name__ == '__main__':
    lv = Live()
    lv.run()
