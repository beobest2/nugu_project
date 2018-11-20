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

        @app.route("/Watcher/initAction", methods=["POST"])
        def watcher_init_action():
            """ 사용자 발화시 최초 접속하는 부분 """
            print("[initAction] : {}".format(request.get_json()))
            print("============================================")

            return json.dumps(request.get_json())

        @app.route("/Watcher/answer.exist", methods=["POST"])
        def watcher_answer_exist():
            """ 'answer.exist' Action으로 들어온 질문을 처리하는 함수 """

            method = request.method
            if method != "POST":
                rtn = {"result" : False, "message": "not supported method [%s]" % method}

            try:
                json_data = request.get_json()
                print("[answer.exist] json_data : {}".format(json_data))
            except:
                json_data = None

            """ FIXME :  disappear_time에 watched가 마지막으로 존재했던 시간을 할당
            1. watched는 json_data['action']['parameters']['watched']['value'] 에 담겨있음
            2. disappear_time은 현재 화면에서 watched가 인식되지 않을 경우에만 할당해주면 됨
            3. 현재 화면에서 인식 가능한 경우에는 disappear_time 에 0 을 주면 됨
            4. 옵션으로 'hour_', 'min_', 'now_' 값이 올 수도 있음
                - json_data['action']['parameters']['hour_'] (min_, now_) 가 존재하면 옵션이 있는 경우임
                - hour_, min_ ,now_ 중 아무것도 존재하지 않으면 옵션을 주지 않은 것임
                - hour_ 의 경우 앞에 'D.'이 붙어서 옴 (ex. D.1, D.2)
                - 옵션이 있는 경우 현재 시간에서 해당 시간을 뺀 시점에 'watched'가 있었는지 확인하면됨
                - 옵션이 없는 경우는 현재 화면을 기준으로 판단하면 됨
                - 즉, 옵션이 없는 경우와 옵션이 '지금'인 경우는 동일하게 처리
            5. 현재는 테스트용으로 임의의 값을 설정해놓았음 """

            # 아래는 테스트용임
            import random
            disappear_time = random.choice([0, "13시 13분"])  # 0인 경우는 인식 가능, 후자는 인식 불가시 사라진 시간

            """ FIXME : watched가 'UNKNOWN' 으로 전달되었을때 처리
            - 처음 보는 객체가 발견됐다면, unknown에 해당 객체명 할당 
            - disappear_time은 1을 할당  
            - 처음 보는 객체가 없다면 disappear_time에 -1을 할당 """

            unknown = 0  # unknown default값. 뭐가 오든 상관없음
            watched = json_data['action']['parameters']['watched']['value']
            if watched == "UNKNOWN":
                disappear_time = random.choice([1, -1])  # 특이객체 있을경우 1, 없을 경우 -1
                if disappear_time == 1:  # 특이객체 있을경우 처리, 현재는 랜덤값을 넣어놓았음
                    unknown = random.choice(["귀신", "도꺠비", "팀장님"])

            """ disappear_time 경우의수
            -1 : 이상한사람에 대한 질문시, 이상한사람이 없을경우
            1 : 이상한사람에 대한 질문시, 이상한사람이 있을경우
            0 : 미리 등록해놓은 사람에 대한 질문시, 자리에 있을 경우
            실제값 : 미리 등록해놓은 사람에 대한 질문시, 사라진 시간 ("xx시 xx분" 형태로 할당)
            """

            rtn = {
                    "version": "2.0",
                    "resultCode": "OK",
                    "output": {
                        "result": True,
                        "unknown": unknown,
                        "disappear_time": disappear_time
                    }
                }

            # return한 값을 우선 nugu play builder에서 체크한 후, not_exist로 라우팅 할지말지 결정한다.
            # disappear_time이 존재하지 않는다면 현재 사용자가 있는 것이므로, not_exist 라우터를 타지 않는다.
            print("[answer.exist] json.dumps(rtn) : {}".format(json.dumps(rtn)))
            print("============================================")
            return json.dumps(rtn)

        @app.route("/Watcher/not_exist", methods=["POST"])
        def watcher_not_exist():
            """ watcher_answer_exist 함수의 분석 결과, 사용자가 존재하지 않으면 처리하는 함수 """

            method = request.method
            if method != "POST":
                rtn = {"result" : False, "message": "not supported method [%s]" % method}

            try:
                json_data = request.get_json()
                print("[not_exist] json_data : {}".format(json_data))
            except:
                json_data = None

            disappear_time = json_data['action']['parameters']['disappear_time']['value']  # "xx시 xx분" 형태로 할당
            rtn = {
                    "version": "2.0",
                    "resultCode": "OK",
                    "output": {
                        "result": True,
                        "disappear_time": disappear_time
                    }
                }

            print("[not_exist] json.dumps(rtn) : {}".format(json.dumps(rtn)))
            return json.dumps(rtn)


        @app.route("/Watcher/answer.capture", methods=["POST"])
        def watcher_answer_capture():
            """ watcher_answer_exist 함수의 분석 결과, 사용자가 존재하지 않으면 처리하는 함수 """

            method = request.method
            if method != "POST":
                rtn = {"result": False, "message": "not supported method [%s]" % method}

            try:
                json_data = request.get_json()
                print("[answer_capture] json_data : {}".format(json_data))
            except:
                json_data = None

            """ FIXME : 사진을 이메일로 전송한다.
            1. 이메일 주소는 임의로 등록한 값이며, 서버단에서 설정파일등을 통해 처리
            2. 옵션으로 'hour', 'min', 'now' 값이 올 수도 있음
                - json_data['action']['parameters']['hour'] (min, now) 가 존재하면 옵션이 있는 경우임
                - hour, min ,now 중 아무것도 존재하지 않으면 옵션을 주지 않은 것임
                - 옵션이 있는 경우 현재로부터 해당 시간만큼 뺀 시점의 사진을 메일로 보내주면 됨
                - 옵션이 없는 경우는 현재 사진을 찍어서 메일로 보내주면 됨
                - 즉, 옵션이 없는 경우와 옵션이 '지금'인 경우는 동일하게 처리
            3. 메일을 보낸후 resultCode를 OK로 보내주면 된다.
            4. 메일 전송이 실패한 경우는 아직 처리하지 않는다.
            """
            rtn = {
                "resultCode": "OK"
            }

            print("[answer_capture] json.dumps(rtn) : {}".format(json.dumps(rtn)))
            print("============================================")
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
