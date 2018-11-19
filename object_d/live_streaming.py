# live_streaming.py

from flask import Flask, render_template, Response, request, jsonify
import json
import object_detector
import camera
import cv2
import face_recog
import datetime
import time
import Socket

class Live():
    def __init__(self):
        self.host = '0.0.0.0'
        self.port = 5050
        self.debug = True
        self.current_time = 30 # second
        self.current_buffer = []
        self.app = Flask(__name__)
        self.init_flask(self.app)

    def init_flask(self, app):

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/video_feed')
        def video_feed():
            return Response(gen(face_recog.FaceRecog()),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        def gen(fr):
            detector = object_detector.ObjectDetector('ssd_mobilenet_v1_coco_2017_11_17')
            #detector = ObjectDetector('mask_rcnn_inception_v2_coco_2018_01_28')
            #detector = ObjectDetector('pet', label_file='data/pet_label_map.pbtxt')

            #cam = camera.VideoCamera()
            retry_cnt = 0
            while True:
                try:
                    frame, face_result_list = fr.get_frame_live()
                    #frame = cam.get_frame()
                    frame, obj_detect_dict = detector.detect_objects_live(frame)
                    self.buffer_handle(face_result_list, obj_detect_dict)

                    ret, jpg = cv2.imencode('.jpg', frame)
                    jpg_bytes = jpg.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')
                except:
                    try:
                        time.sleep(1)
                        if retry_cnt > 5:
                            break
                        fr = face_recog.FaceRecog()
                    except:
                        retry_cnt += 1

        @app.route("/health", methods=["GET"])
        def health_check():
            method = request.method
            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": True,
                      "message": "health chkeck success"
                    }
                }
            return json.dumps(rtn)

        @app.route("/show", methods=["GET"])
        def show():
            method = request.method
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

        @app.route("/Watcher", methods=["POST"])
        def watcher():
            method = request.method
            if method != "POST":
                rtn = {"result" : False, "message": "not supported method [%s]" % method}

            try:
                json_data = request.get_json()
            except:
                json_data = None
            database = ""
            if json_data is not None:
                database = json_data['database']

            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": True,
                      "disappear_time": 10
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

if __name__ == '__main__':
    lv = Live()
    lv.run()
