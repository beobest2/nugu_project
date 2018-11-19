# live_streaming.py

from flask import Flask, render_template, Response, request, jsonify
import json
import object_detector
import camera
import cv2
import face_recog
import datetime
import time
import struct
import pickle
import socket
import Socket

class Live():
    def __init__(self):
        self.host = '0.0.0.0'
        self.port = 5060

        self.video_host = '127.0.0.1'
        self.video_port = 5051

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
            try:
                resp = Response(gen(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
            except:
                return "video server connection fail"
            return resp

        def gen():
            host = '127.0.0.1'
            port = 5051
            clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            clientsocket.connect((host, port))

            data = ""
            payload_size = struct.calcsize("H")
            while True:
                while len(data) < payload_size:
                    data += clientsocket.recv(4096)
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("H", packed_msg_size)[0]
                while len(data) < msg_size:
                    data += clientsocket.recv(4096)
                frame_data = data[:msg_size]
                data = data[msg_size:]
                frame=pickle.loads(frame_data)

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
                      "message": "health check success"
                    }
                }
            return json.dumps(rtn)

        @app.route("/show", methods=["GET"])
        def show():
            method = request.method
            s = Socket.Socket()
            s.Connect(self.video_host, self.video_port)
            print(s.ReadMessage())
            s.SendMessage(b"SHOW_CURRENT 0\r\n")
            read_msg = s.ReadMessage()
            s.SendMessage(b"QUIT 0\r\n")
            s.close()
            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": True,
                      "message": str(read_msg)
                    }
                }
            return json.dumps(rtn)

        @app.route("/Watcher/answer.exist", methods=["POST"])
        def watcher_answer_exist():
            """ 'answer.exist' Action으로 들어온 질문을 처리하는 함수 """

            method = request.method
            if method != "POST":
                rtn = {"result" : False, "message": "not supported method [%s]" % method}

            try:
                json_data = request.get_json()
            except:
                json_data = None

            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": True,
                      "disappear_time": 10
                    }
                }

            # return한 값을 우선 nugu play builder에서 체크한 후, not_exist로 라우팅 할지말지 결정한다.
            # disappear_time이 존재하지 않는다면 현재 사용자가 있는 것이므로, not_exist 라우터를 타지 않는다.   
            return json.dumps(rtn)

        @app.route("/Watcher/not_exist", methods=["POST"])
        def watcher_not_exist():
            """ watcher_answer_exist 함수의 분석 결과, 사용자가 존재하지 않으면 처리하는 함수 """

            method = request.method
            if method != "POST":
                rtn = {"result" : False, "message": "not supported method [%s]" % method}

            try:
                json_data = request.get_json()
            except:
                json_data = None

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
