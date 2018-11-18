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

            cam = camera.VideoCamera()

            while True:
                frame, face_names = fr.get_frame_live()
                self.buffer_handle_face(face_names)
                #frame = cam.get_frame()
                frame = detector.detect_objects(frame)

                ret, jpg = cv2.imencode('.jpg', frame)
                jpg_bytes = jpg.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n\r\n')

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

        @app.route("/test", methods=["POST"])
        def test():
            method = request.method
            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": True,
                      "disappear_time": 10
                    }
                }
            return json.dumps(rtn)


    def buffer_handle_face(self, face_names):
        now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        for item in face_names:
            self.BUFFER["face_names"].append((face_names, now_str))
        if len(self.BUFFER) > 10:
            self.empty_buffer()
        #print(self.BUFFER)

    def empty_buffer(self):
        self.BUFFER = {"face_names": []}

    def run(self):
        self.app.run(host=self.host, debug=self.debug)

if __name__ == '__main__':
    lv = Live()
    lv.run()
