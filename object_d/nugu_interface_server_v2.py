# live_streaming.py

from flask import Flask, render_template, Response, request, jsonify
import json
import datetime
import time
import pymysql
import random
import server_conf

class Live():
    def __init__(self):
        self.proxy_host = server_conf.proxy_host
        self.proxy_port = server_conf.proxy_port
        self.proxy_debug = True
        self.app = Flask(__name__)
        self.init_flask(self.app)

        # 현재 시간이라고 볼수있는 시간 범위 - 최근 레코드 개수
        self.now_time_range = server_conf.now_time_range
        # 과거 질문시 +- 10분 까지 조회
        self.past_range = server_conf.past_range

        # 접속할 mysql sever connection info
        self.mysql_host = server_conf.mysql_host
        self.mysql_user = server_conf.mysql_user
        self.mysql_password = server_conf.mysql_password
        self.mysql_db = server_conf.mysql_db
        self.mysql_table = server_conf.mysql_table
        self.mysql_img_call_table = server_conf.mysql_img_call_table
        self.mysql_img_file_table = server_conf.mysql_img_file_table
        """
        DB : testcam01
        TABLE : testcam01
        date | class | corr | move

        TABLE : imgcall
        call_date | time <-- 10분 단위로 요청, now일 경우 0
        """

        # LABEL dict setting
        self.label_dict = server_conf.label_dict
        self.known_dict = {}
        self.watched_dict = {}
        for item in self.label_dict.keys():
            self.known_dict[item] = self.label_dict[item][0]
        for item in self.label_dict.keys():
            for watched in self.label_dict[item]:
                self.watched_dict[watched] = item


    def _mysql_connection_check(self):
        # health check
        return_val = True
        try:
            conn = pymysql.connect(host=self.mysql_host, user=self.mysql_user,
                    password=self.mysql_password, db=self.mysql_db, charset='utf8')
            curs = conn.cursor(pymysql.cursors.DictCursor)
        except Exception as err:
            print(err)
            return_val = False
        finally:
            try:
                conn.close()
            except:
                return_val = False
                pass
        return return_val


    def _mysql_dml(self, sql, val=None):
        # INSERT, UPDATE, DELETE
        try:
            conn = pymysql.connect(host=self.mysql_host, user=self.mysql_user,
                    password=self.mysql_password, db=self.mysql_db, charset='utf8')
            curs = conn.cursor(pymysql.cursors.DictCursor)
            if val is None:
                curs.execute(sql)
            else:
                curs.execute(sql, val)
            conn.commit()
        except Exception as err:
            print(err)
        finally:
            try:
                conn.close()
            except:
                pass

    def _mysql_select(self, sql):
        # SELECT
        # return rows
        rows = []
        try:
            conn = pymysql.connect(host=self.mysql_host, user=self.mysql_user,
                    password=self.mysql_password, db=self.mysql_db, charset='utf8')
            curs = conn.cursor(pymysql.cursors.DictCursor)
            curs.execute(sql)
            rows = curs.fetchall()
        except Exception as err:
            print(err)
        finally:
            try:
                conn.close()
            except:
                pass
        return rows

    def last_check_db(self, target):
        sql = "SELECT DATE FROM %s WHERE " % self.mysql_table
        sql += " CLASS = '%s' ORDER BY DATE DESC LIMIT 1" % target
        date = None
        rows = self._mysql_select(sql)
        if len(rows) >= 1:
            date = rows[0]["DATE"]
        return date

    def LAST_SHOW(self, target):
        print("!!!!!!! LAST SHOW : ", target)
        rtn_str = None
        last_date_str = self.last_check_db(target)
        print("last_date_str: ", last_date_str)
        if last_date_str is None:
            pass
        else:
            # calculate time delta
            last_date = datetime.datetime.strptime(str(last_date_str), '%m%d%H%M%S')
            time_delta = datetime.datetime.now() - last_date
            h = 0
            m = 0
            h, rem = divmod(time_delta.seconds, 3600)
            m, s = divmod(rem, 60)
            rtn_str = "%s,%s" % (h, m)
        print("rtn_str ::::: ", rtn_str)
        return rtn_str

    def SHOW_CURRENT(self):
        now_date = datetime.datetime.now()
        now_min_date = now_date - datetime.timedelta(seconds=self.now_time_range)
        now_str = now_date.strftime("%m%d%H%M%S")
        now_min_str = now_min_date.strftime("%m%d%H%M%S")
        sql = "SELECT CLASS FROM %s WHERE " % self.mysql_table
        sql += " DATE >= %s ORDER BY DATE DESC LIMIT 3" % int(now_min_str)
        print("show current  sql: ", sql)
        rows = self._mysql_select(sql)
        tmp_set = set([])
        for item in rows:
            tmp_set.add(item["CLASS"])
        rtn_list = []
        for item in tmp_set:
            rtn_list.append(item)
        return rtn_list

    def SHOW_PAST(self, past_min):
        now_date = datetime.datetime.now()
        past_min_date = now_date - datetime.timedelta(minutes=past_min+self.past_range)
        past_max_date = now_date - datetime.timedelta(minutes=past_min-self.past_range)
        past_min_str = past_min_date.strftime("%m%d%H%M%S")
        past_max_str = past_max_date.strftime("%m%d%H%M%S")
        sql = "SELECT CLASS FROM %s WHERE " % self.mysql_table
        sql += " DATE >= %s AND DATE <= %s ORDER BY DATE" % (int(past_min_str), int(past_max_str))
        rows = self._mysql_select(sql)
        tmp_set = set([])
        for item in rows:
            tmp_set.add(item["CLASS"])
        rtn_list = []
        for item in tmp_set:
            rtn_list.append(item)
        return rtn_list

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
            if not self._mysql_connection_check():
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
            rtn_list = self.SHOW_CURRENT()
            rtn_str = ",".join(rtn_list)
            rtn = {
                    "version": "2.0",
                    "resultCode": "200 OK",
                    "output": {
                      "result": True,
                      "message": rtn_str
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

            # set_default_value
            with_in_a_minute = 0
            is_exist = 0
            action_type = None
            disappear_time = -99 # default
            watched = ""
            try:
                watched = json_data['action']['parameters']['watched']['value']
            except:
                pass

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

            k_list = json_data['action']['parameters'].keys()
            if "hour_" in k_list or "min_" in k_list:
                hour_ = 0
                min_ = 0
                if "hour_" in k_list:
                    if json_data['action']['parameters']['hour_']["value"].startswith("D"):
                        hour_ = int(json_data['action']['parameters']['hour_']["value"].split(".")[1])
                    else:
                        hour_ = int(json_data['action']['parameters']['hour_']["value"])
                if "min_" in k_list:
                    min_ = int(json_data['action']['parameters']['min_']["value"])
                print("HOUR : ", hour_)
                print("MIN : ", min_)
                total_past_min = (hour_ * 60) + min_
                # 과거
                if watched.upper() == "UNKNOWN":
                    """ FIXME : watched가 'UNKNOWN' 으로 전달되었을때 처리
                    - 처음 보는 객체가 발견됐다면, unknown에 해당 객체명 할당 
                    - disappear_time은 1을 할당  
                    - 처음 보는 객체가 없다면 disappear_time에 -1을 할당 """
                    rtn_list = self.SHOW_PAST(total_past_min)
                    print("rtnl;ist ::::: {}".format(rtn_list))
                    if "Unknown" in rtn_list:
                        # 낯선 사람 과거 시점에 존재
                        action_type = "unknown_past"
                        is_exist = 1
                        watched = random.choice(["수상한사람", "모르는사람", "처음보는사람"])
                    else:
                        # 낯선 사람 없음
                        action_type = "unknown_past"
                        is_exist = 0
                elif watched.upper() == "ALL":
                    """ FIXME : watched가 'ALL'로 전달되었을때 처리
                    - 해당 시점에 관측된 모든 객체를 'all' 에 문자열 형태로 담아서 리턴
                        - ex) 현우, 해준, 컴퓨터, 강아지가 있을 경우
                        - detected_list = ["현우","해준","컴퓨터","강아지"]
                        - all = ",".join(detected_list)
                    - disappear_time은 10을 할당
                     """
                    rtn_list = self.SHOW_PAST(total_past_min)
                    if len(rtn_list) > 0:
                        detected_list = self.detected_list_match(rtn_list)
                        watched = ",".join(detected_list)
                        action_type = "all_past"
                        is_exist = 1
                    else:
                        action_type = "all_past"
                        is_exist = 0
                else:
                    # 특정 객체 질문
                    target = watched
                    if watched in self.watched_dict.keys():
                        target = self.watched_dict[watched]

                    rtn_list = self.SHOW_PAST(total_past_min)
                    if target in rtn_list:
                        # 지금 존재
                        action_type = "one_past"
                        is_exist = 1
                    else:
                        # 지금 존재하지 않을 경우 언제 사라졌는지 조사
                        action_type = "one_past"
                        is_exist = 0
            else:
                # 현재
                if watched.upper() == "UNKNOWN":
                    """ FIXME : watched가 'UNKNOWN' 으로 전달되었을때 처리
                    - 처음 보는 객체가 발견됐다면, unknown에 해당 객체명 할당 
                    - disappear_time은 1을 할당  
                    - 처음 보는 객체가 없다면 disappear_time에 -1을 할당 """
                    rtn_list = self.SHOW_CURRENT()
                    if "Unknown" in rtn_list:
                        # 낯선 사람 지금 존재
                        action_type = "unknown_now"
                        is_exist = 1
                        watched = random.choice(["수상한사람", "모르는사람", "처음보는사람"])
                    else:
                        # 낯선 사람 없음
                        action_type = "unknown_now"
                        is_exist = 0
                        read_msg = self.LAST_SHOW("UNKNOWN")
                        if read_msg == "0,0":
                            action_type = "unknown_now"
                            is_exist = 0
                            with_in_a_minute = 1
                            watched = random.choice(["수상한사람", "모르는사람", "처음보는사람"])
                        elif read_msg is None:
                            action_type = "unknown_now"
                            is_exist = -1
                        else:
                            print("read_msg: ", read_msg)
                            rtn_list = read_msg.strip().split(",")
                            print("rtn_list: ", rtn_list)
                            hour_ = int(rtn_list[0])
                            min_ = int(rtn_list[1])
                            watched = random.choice(["수상한사람", "모르는사람", "처음보는사람"])
                            if hour_ == 0:
                                disappear_time = "%d분" % min_
                            else:
                                disappear_time = "%d시%d분" % (hour_, min_)
                elif watched.upper() == "ALL":
                    """ FIXME : watched가 'ALL'로 전달되었을때 처리
                    - 해당 시점에 관측된 모든 객체를 'all' 에 문자열 형태로 담아서 리턴
                        - ex) 현우, 해준, 컴퓨터, 강아지가 있을 경우
                        - detected_list = ["현우","해준","컴퓨터","강아지"]
                        - all = ",".join(detected_list)
                    - disappear_time은 10을 할당
                     """
                    rtn_list = self.SHOW_CURRENT()
                    if len(rtn_list) > 0:
                        detected_list = self.detected_list_match(rtn_list)
                        watched = ",".join(detected_list)
                        action_type = "all_now"
                        is_exist = 1
                    else:
                        action_type = "all_now"
                        is_exist = 0
                else:
                    # 특정 객체 질문
                    target = watched
                    if watched in self.watched_dict.keys():
                        target = self.watched_dict[watched]

                    rtn_list = self.SHOW_CURRENT()
                    if target in rtn_list:
                        # 지금 존재
                        action_type = "one_now"
                        is_exist = 1
                    else:
                        # 지금 존재하지 않을 경우 언제 사라졌는지 조사
                        action_type = "one_now"
                        is_exist = 0
                        read_msg = self.LAST_SHOW(target)
                        if read_msg == "0,0":
                            action_type = "one_now"
                            is_exist = 0
                            with_in_a_minute = 1
                        elif read_msg is None:
                            action_type = "one_now"
                            is_exist = -1
                        else:
                            print("read_msg: ", read_msg)
                            rtn_list = read_msg.strip().split(",")
                            print("rtn_list: ", rtn_list)
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
                        "watched": watched,
                        "disappear_time": disappear_time,
                        "action_type" : action_type,
                        "is_exist" : is_exist,
                        "with_in_a_minute" : with_in_a_minute
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

            total_past_min = 0
            k_list = json_data['action']['parameters'].keys()
            if "hour" in k_list or "min" in k_list:
                hour_ = 0
                min_ = 0
                if "hour" in k_list:
                    hour_ = int(json_data['action']['parameters']['hour']["value"].split(".")[1])
                if "min" in k_list:
                    min_ = int(json_data['action']['parameters']['min']["value"])
                print("HOUR : ", hour_)
                print("MIN : ", min_)
                total_past_min = (hour_ * 60) + min_

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
            request_time = total_past_min
            now_date = datetime.datetime.now()
            now_str = now_date.strftime("%m%d%H%M%S")

            sql = "INSERT INTO %s (DATE_CALL, TIME) VALUES " % self.mysql_img_call_table
            sql += " (%s, %s)"
            self._mysql_dml(sql, (int(now_str), request_time))
            rtn = {
                "resultCode": "OK"
            }

            print("[answer_capture] json.dumps(rtn) : {}".format(json.dumps(rtn)))
            print("============================================")
            return json.dumps(rtn)

    def run(self):
        self.app.run(host=self.proxy_host, port=self.proxy_port, debug=self.proxy_debug)

if __name__ == '__main__':
    lv = Live()
    lv.run()
