# live_streaming.py

from flask import Flask, render_template, Response, request, jsonify
import json
import object_detector
import camera
import cv2
import face_recog
import datetime

class Live():
    def __init__(self):
        self.host = '0.0.0.0'
        self.debug = True
        self.BUFFER = {"face_names": []}
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
                    fr = face_recog.FaceRecog() 

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


    def buffer_handle(self, face_result_list, obj_detect_dict):
        for face_result in face_result_list:
            face_corr = face_result[0]
            face_name = face_result[1]
        for item in obj_detect_dict:
            try:
                obj_corr = item
                class_str = obj_detect_dict[item][0]
                class_list = class_str.split()
                obj_class = class_list[0][:-1]
                obj_score = int(class_list[1][:-1])
            except:
                pass
        #self.BUFFER.append(face_name)
        #print(face_name)
        #print(obj_class)
        now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        #print(now_str)
        #for item in face_names:
        #    self.BUFFER["face_names"].append((face_names, now_str))
        if len(self.BUFFER) > 100:
            del self.BUFFER[0]

    def run(self):
        self.app.run(host=self.host, debug=self.debug)

if __name__ == '__main__':
    lv = Live()
    lv.run()
