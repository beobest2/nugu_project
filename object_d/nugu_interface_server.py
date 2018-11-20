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

        self.known_dict = {"HYUNWOO" : "현우", "HAEJOON" : "해준", "person" : "사람", "dog" : "강아지", "cat" : "고양이"}
        self.watched_dict = {"현우" : "HYUNWOO", "해준" : "HAEJOON", "사람" : "person", "강아지" :
                "dog", "고양이" : "cat", "야옹이" : "cat", "냐옹이" : "cat", "멍멍이" : "cat"}

    def communicate_video(self, cmd):
        read_msg = "-ERR fail to connect video server\r\n"
        try:
            s = Socket.Socket()
            s.Connect(self.video_host, self.video_port)
            s.ReadMessage() # welcome msg
            s.SendMessage(cmd)
            read_msg = s.ReadMessage()
            s.SendMessage(b"QUIT 0\r\n")
        except Exception as err:
            print("??", err)
            pass
        finally:
            try:
                s.close()
            except:
                pass
        rtn_val = ""
        if read_msg[0]:
            rtn_val = read_msg[1]
        return str(rtn_val)

    def detected_list_match(self, rtn_list):
        for idx, item in enumerate(rtn_list):
            if item in self.known_dict.keys():
                rtn_list[idx] = self.known_dict[item]
        return rtn_list

    def init_flask(self, app):
        @app.route('/')
        def index():
            rtn_msg = "health check success"
            rtn_bool = True

            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": rtn_bool,
                      "message": rtn_msg
                    }
                }
            return json.dumps(rtn)

        @app.route("/health", methods=["GET"])
        def health_check():
            """
            health check : 시스템  정상 연결 확인
            """
            method = request.method
            rtn_msg = "health check success"
            rtn_bool = True
            read_msg = ""
            read_msg = self.communicate_video(b"HEALTH_CHECK 0\r\n")
            if read_msg.strip() != "+OK 30":
                rtn_bool = False
                rtn_msg = "video server connection fail"

            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": rtn_bool,
                      "message": rtn_msg
                    }
                }
            return json.dumps(rtn)

        @app.route("/show", methods=["GET"])
        def show():
            """
            테스트용 : 현재 검출된 객체들의 목록 반환
            """
            method = request.method
            read_msg = self.communicate_video(b"SHOW_CURRENT 0\r\n")
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

            # disappear value
            DEFAULT = -99
            UNKNOWN_NOT_EXIST = -1
            UNKNOWN_EXIST = 1
            TARGET_EXIST = 0
            TARGET_ALL = 10
            TARGET_NOT_EXIST = "%d시 %d분"

            try:
                json_data = request.get_json()
                print("[answer.exist] json_data : {}".format(json_data))
            except:
                json_data = None

            all = None
            unknown = 0  # unknown default값. 뭐가 오든 상관없음
            disappear_time = DEFAULT
            watched = json_data['action']['parameters']['watched']['value']

            """ FIXME :  disappear_time에 watched가 마지막으로 존재했던 시간을 할당
            질문 : 낯선 사람 있니 / 철수 있니
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

            k_list = json_data['action']['parameters'].keys()
            if "hour_" in k_list or "min_" in k_list:
                # 과거
                hour_val = None
                min_val = None
                if "hour_" in k_list:
                    hour_val = json_data['action']['parameters']['hour_']
                if "min_" in k_list:
                    mon_val = json_data['action']['parameters']['min_']

                if watched == "UNKNOWN":
                    """ FIXME : watched가 'UNKNOWN' 으로 전달되었을때 처리
                    - 처음 보는 객체가 발견됐다면, unknown에 해당 객체명 할당 
                    - disappear_time은 1을 할당  
                    - 처음 보는 객체가 없다면 disappear_time에 -1을 할당 """
                    read_msg = self.communicate_video(b"SHOW_PAST 0\r\n")
                    rtn_list = read_msg.split(",")
                    if "UNKNOWN" in rtn_list:
                        # 낯선 사람 지금 존재
                        disappear_time = UNKNOWN_EXIST
                    else:
                        disappear_time = UNKNOWN_NOT_EXIST
                        unknown = random.choice(["수상한사람", "모르는사람", "처음보는사람"])

                elif watched == "ALL":
                    """ FIXME : watched가 'ALL'로 전달되었을때 처리
                    - 해당 시점에 관측된 모든 객체를 'all' 에 문자열 형태로 담아서 리턴
                        - ex) 현우, 해준, 컴퓨터, 강아지가 있을 경우
                        - detected_list = ["현우","해준","컴퓨터","강아지"]
                        - all = ",".join(detected_list)
                    - disappear_time은 10을 할당
                     """
                    read_msg = self.communicate_video(b"SHOW_PAST 0\r\n")
                    rtn_list = read_msg.split(",")
                    detected_list = self.detected_list_match(rtn_list)
                    all = ",".join(detected_list)
                    disappear_time = TARGET_ALL
                else:
                    # 특정 객체 질문
                    target = watched
                    if watched in self.watched_dict.keys():
                        target = self.watched_dict[watched]

                    read_msg = self.communicate_video(b"SHOW_PAST 0\r\n")
                    rtn_list = read_msg.split(",")
                    if target in rtn_list:
                        # 지금 존재
                        disappear_time = TARGET_EXIST
                    else:
                        disappear_time = TARGET_NOT_EXIST
                        last_cmd = "LAST_SHOW %s\r\n" % target
                        last_cmd_b = bytes(last_cmd, 'utf-8') 
                        read_msg = self.communicate_video(last_cmd_b)
                        if len(read_msg) == 0:
                            # db에 없다
                            pass
                        else:
                            rtn_list = read_msg.strip().split(",")
                            hour_ = int(rtn_list[0])
                            min_ = int(rtn_list[1])
                            if hour_ == 0:
                                disappear_time = "%d분" % min_
                            else:
                                disappear_time = "%d시%d분" % (hour_, min_)
            else:
                # 현재
                if watched == "UNKNOWN":
                    """ FIXME : watched가 'UNKNOWN' 으로 전달되었을때 처리
                    - 처음 보는 객체가 발견됐다면, unknown에 해당 객체명 할당 
                    - disappear_time은 1을 할당  
                    - 처음 보는 객체가 없다면 disappear_time에 -1을 할당 """
                    read_msg = self.communicate_video(b"SHOW_CURRENT 0\r\n")
                    rtn_list = read_msg.split(",")
                    if "UNKNOWN" in rtn_list:
                        # 낯선 사람 지금 존재
                        disappear_time = UNKNOWN_EXIST
                    else:
                        disappear_time = UNKNOWN_NOT_EXIST
                        unknown = random.choice(["수상한사람", "모르는사람", "처음보는사람"])

                elif watched == "ALL":
                    """ FIXME : watched가 'ALL'로 전달되었을때 처리
                    - 해당 시점에 관측된 모든 객체를 'all' 에 문자열 형태로 담아서 리턴
                        - ex) 현우, 해준, 컴퓨터, 강아지가 있을 경우
                        - detected_list = ["현우","해준","컴퓨터","강아지"]
                        - all = ",".join(detected_list)
                    - disappear_time은 10을 할당
                     """
                    read_msg = self.communicate_video(b"SHOW_CURRENT 0\r\n")
                    rtn_list = read_msg.split(",")
                    detected_list = self.detected_list_match(rtn_list)
                    all = ",".join(detected_list)
                    disappear_time = TARGET_ALL
                else:
                    # 특정 객체 질문
                    target = watched
                    if watched in self.watched_dict.keys():
                        target = self.watched_dict[watched]

                    read_msg = self.communicate_video(b"SHOW_CURRENT 0\r\n")
                    rtn_list = read_msg.split(",")
                    if target in rtn_list:
                        # 지금 존재
                        disappear_time = TARGET_EXIST
                    else:
                        disappear_time = TARGET_NOT_EXIST
                        last_cmd = "LAST_SHOW %s\r\n" % target
                        last_cmd_b = bytes(last_cmd, 'utf-8') 
                        read_msg = self.communicate_video(last_cmd_b)
                        if len(read_msg) == 0:
                            # db에 없다
                            pass
                        else:
                            rtn_list = read_msg.strip().split(",")
                            hour_ = int(rtn_list[0])
                            min_ = int(rtn_list[1])
                            if hour_ == 0:
                                disappear_time = "%d분" % min_
                            else:
                                disappear_time = "%d시%d분" % (hour_, min_)

            rtn = {
                    "version": "2.0",
                    "resultCode": "OK",
                    "output": {
                        "result": True,
                        "all": all,
                        "unknown": unknown,
                        "disappear_time": disappear_time
                    }
                }

            print("[answer.exist] json.dumps(rtn) : {}".format(json.dumps(rtn)))
            print("============================================")
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

    def run(self):
        self.app.run(host=self.host, port=self.port, debug=self.debug)

if __name__ == '__main__':
    lv = Live()
    lv.run()
