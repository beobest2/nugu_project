# 2018.11.21

import cv2
import os
import numpy as np
import tensorflow as tf
import tarfile
import six.moves.urllib as urllib
import face_recognition
import threading
import datetime
import time
import queue
import pymysql
import server_conf
import smtplib
import os
import re
import logging
import configparser

from email.message import EmailMessage
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops


class VideoRun():
    def __init__(self):
        # 접속할 mysql sever connection info
        self.mysql_host = server_conf.mysql_host
        self.mysql_user = server_conf.mysql_user
        self.mysql_password = server_conf.mysql_password
        self.mysql_db = server_conf.mysql_db
        self.mysql_table = server_conf.mysql_table
        self.mysql_img_call_table = server_conf.mysql_img_call_table
        self.mysql_img_file_table = server_conf.mysql_img_file_table
        self.user_email = server_conf.user_email
        """
        DB : testcam01
        TABLE : testcam01
        date | class | corr | move

        TABLE : imgcall
        call_date | time <-- 10분 단위로 요청, now일 경우 0
        """

        # frame 정보 관리 메모리
        self.frame = None
        self.frame_cnt = 0
        self.now_date = datetime.datetime.now()
        self.buffer_list = []

        # db 입력 간격(5초)
        self.db_insert_term = server_conf.db_insert_term
        # db file 저장 시간 최대 범위(2 시간)
        self.max_db_date = server_conf.max_db_date

        # img file 관리 큐
        self.img_file_queue = queue.Queue()
        # img file 저장 최대 개수 (1000장)
        self.max_img_file_cnt = server_conf.max_img_file_cnt
        # 이미지 파일 저장 경로
        self.imwrite_path = server_conf.imwrite_path
        # 이미지 파일 저장 시간 간격(10분 마다)
        self.img_write_gap = server_conf.img_write_gap

        # 비디오 서버 재연결 시도 횟수
        self.retry_cnt_max = 5

        # 버퍼에 데이터 담는 시간 간격
        self.buffer_write_gap = server_conf.buffer_write_gap

        # logger
        self.logger = self.get_logger("video_server")

    def run_video(self):
        # 얼굴 인식 모델 호출
        face_recog_m = FaceRecog()
        # 객체 인식 모델 호출
        detector = ObjectDetector('ssd_mobilenet_v1_coco_2017_11_17')
        self.logger.debug(" ** known faces: " + str( face_recog_m.known_face_names))

        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.

        def func(face_recog_m, detector):
            # camera 호출
            camera = VideoCamera(server_conf.camera_source)

            retry_cnt = 0

            insert_flag = False
            insert_priv_time = time.time()

            imwrite_flag = False
            img_write_priv_time = time.time()

            bufferwrite_flag = False
            buffer_write_priv_time = time.time()

            while True:
                frame = camera.get_frame()
                now_date = datetime.datetime.now()
                self.now_date = now_date
                now_str = now_date.strftime("%m%d%H%M%S")

                curr_time = time.time()
                # db 입력 간격 측정
                if curr_time - insert_priv_time >= self.db_insert_term:
                    insert_priv_time = curr_time
                    insert_flag = True
                else:
                    insert_flag = False

                # 이미지 저장 간격 측정
                if curr_time - img_write_priv_time >= self.img_write_gap:
                    img_write_priv_time = curr_time
                    imwrite_flag = True
                else:
                    imwrite_flag = False

                # 버퍼 저장 간격 측정
                if curr_time - buffer_write_priv_time >= self.buffer_write_gap:
                    buffer_write_priv_time = curr_time
                    bufferwrite_flag = True
                else:
                    bufferwrite_flag = False

                try:
                    # image write flag check
                    if imwrite_flag:
                        # 10 분마다 이미지파일 쓰기
                        img_write_file = self.imwrite_path + now_str + ".jpg"
                        cv2.imwrite(img_write_file, frame)
                        self.img_file_queue.put(img_write_file)
                        # DB INSERT
                        self.logger.debug("insert img path")
                        sql = "INSERT INTO %s (DATE, FILE_PATH) VALUES " % self.mysql_img_file_table
                        self._mysql_dml(sql + "(%s, %s)", (int(now_str), img_write_file))
                        self.logger.debug("write img: " + str(img_write_file))

                        # 이미지 파일 개수가 1000개 이상일때 삭제
                        if self.img_file_queue.qsize() > self.max_img_file_cnt:
                            del_img_file = self.img_file_queue.get()
                            os.remove(del_img_file)
                            #  DB DELETE
                            self.logger.debug("clean call db")
                            sql = "DELETE FROM %s WHERE FILE_PATH = " % self.mysql_img_file_table
                            self._mysql_dml(sql + " %s", (img_write_file))

                            self.logger.debug("remove img: " + str(del_img_file))

                        # DB에서 2시간 이전의 데이터 삭제
                        self.delete_outdated_data()

                    # 이미지 처리
                    frame, face_result_list = face_recog_m.get_frame_live(frame)
                    frame, obj_detection_dict = detector.detect_objects_live(frame)

                    if insert_flag:
                        # buffer에 축적된 데이터 bulk insert
                        #print("bulk insert")
                        if len(self.buffer_list) > 0:
                            self.bulk_insert_db(self.mysql_table, self.buffer_list)
                        # buffer 초기화
                        self.buffer_list = []

                        # img_call table 확인
                        # DB SELECT
                        sql = "SELECT DATE_CALL, TIME FROM %s" % self.mysql_img_call_table
                        rows = self._mysql_select(sql)
                        #print("check img_call", rows)
                        if len(rows) > 0:
                            # 이미지 전송 요청이 들어옴
                            for item in rows:
                                self.send_img(item["DATE_CALL"], item["TIME"])
                                # call db DELETE
                                self.logger.debug("delete call table")
                                sql = "DELETE FROM %s WHERE DATE_CALL = " % self.mysql_img_call_table
                                self._mysql_dml(sql + " %s", (item["DATE_CALL"]))
                        else:
                            pass
                    # 영상 분석 데이터 처리
                    for face_result in face_result_list:
                        face_corr = face_result[0]
                        face_name = face_result[1]
                        if bufferwrite_flag:
                            self.buffer_list.append((int(now_str), face_name, str(face_corr)))
                    for item in obj_detection_dict:
                        try:
                            obj_corr = item
                            class_str = obj_detection_dict[item][0]
                            obj_str = ""
                            class_list = class_str.split()
                            for idx, st in enumerate(class_list):
                                if idx < len(class_list) - 1:
                                    obj_str += " " + st
                            obj_class = obj_str.strip()[:-1]
                            obj_score = int(class_list[-1][:-1])
                            if bufferwrite_flag:
                                self.buffer_list.append((int(now_str), obj_class, str(obj_corr)))
                        except Exception as err:
                            self.logger.error("OBJ CLASS : " + str(err))
                    self.frame = frame
                    self.frame_cnt += 1
                except Exception as err:
                    self.logger.error(str(err))
                    try:
                        camera = VideoCamera(server_conf.camera_source)
                    except:
                        retry_cnt += 1
                        if retry_cnt > self.retry_cnt_max:
                            break
        # start!
        func(face_recog_m, detector)

        """
        # thread로 돌릴 경우 사용
        th = threading.Thread(target=func, args=(face_recog_m, detector))
        th.daemon = True
        th.start()
        """
        return 0

    def send_img(self, date_call, call_time):
        target_plus_ten_str = ""
        target_minus_ten_str = ""
        # 20181112000000
        try:
            date_call_str = str(date_call)
            # 목표 시간 구하기
            target_date = None
            target_date_str = None
            if call_time == 0:
                target_date_str = date_call_str
                target_date = datetime.datetime.strptime(target_date_str, "%m%d%H%M%S")
            else:
                call_date = datetime.datetime.strptime(date_call_str, "%m%d%H%M%S")
                self.logger.debug("call_date:" + str(call_date))
                target_date = call_date - datetime.timedelta(minutes=call_time)
                self.logger.debug("target_date:" + str(target_date))
                target_date_str = target_date.strftime("%m%d%H%M%S")
            target_plus_ten = target_date + datetime.timedelta(minutes=10)
            target_plus_ten_str = target_plus_ten.strftime("%m%d%H%M%S")
            target_minus_ten = target_date - datetime.timedelta(minutes=10)
            target_minus_ten_str = target_minus_ten.strftime("%m%d%H%M%S")
        except Exception as err:
            self.logger.error("send img : " + str(err))
        # 목표 시간의 +- 10 분 이미지 자료 가져오기
        sql = "SELECT DATE, FILE_PATH FROM %s WHERE " % self.mysql_img_file_table
        rows = self._mysql_select(sql + " DATE <= %s AND DATE >= %s" % (int(target_plus_ten_str), int(target_minus_ten_str)))
        self.logger.debug("get +- 10 min img files: " + str(rows))
        if len(rows) > 0:
            # 값이 있다면
            # FIXME 가장 가까운 시간 찾아서 보내주기
            final_file_path = self.find_simillar_time(int(target_date_str), rows)
            self.logger.debug("Final send file : " + str(final_file_path))
            self.send_email(final_file_path, self.user_email)
            return True
        else:
            self.send_email(None, self.user_email)
            return False

    def find_simillar_time(self, target_date_num, rows):
        min_gap = 10000000000000
        min_path = ""
        for item in rows:
            _date = item["DATE"]
            _file_path = item["FILE_PATH"]
            gap = abs(target_date_num - _date)
            if gap < min_gap:
                min_gap = gap
                min_path = _file_path
        return min_path

    def get_logger(self, logger_name, logging_level=logging.DEBUG):
        """ 로거를 만들어줌 """

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging_level)
        file_handler = logging.FileHandler(f'./logs/{logger_name}.log')
        formatter = logging.Formatter("[%(levelname)s] '%(filename)s' %(asctime)s : %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def send_email(self, file_path, user_email):
        """
        file_path -> "./imgfile/20181212000000.jpg"
        user_email -> ["cleanby@naver.com", "haejoon309@naver.com"]
        """
        conf_path = '../../nugu_conf/nugu.conf'

        config = configparser.ConfigParser()
        config.read(conf_path)

        admin_email = config.get('EMAIL', 'id')
        admin_passwd = config.get('EMAIL', 'passwd')

        # 로거 생성
        logger = self.logger

        msg = EmailMessage()
        msg['From'] = 'Sauron Video Server'
        msg['To'] = user_email
        msg['Subject'] = '[Sauron] 관측 사진 전송'
        # 이메일 내용 셋팅
        if file_path is None:
            msg.set_content("요청하신 시간에 캡쳐된 사진이 없습니다.")
        else:
            date = re.findall('\d+', file_path)[0]
            msg.set_content(f'{date[:2]}월 {date[2:4]}일 {date[4:6]}시 {date[6:8]}분 {date[8:10]}초에 캡쳐된 사진입니다.')
            file = open(file_path, 'rb').read()
            msg.add_attachment(file, maintype='text', subtype='plain', filename="{}.jpg".format(date))

        # 이메일 서버 셋팅
        naver_server = smtplib.SMTP_SSL('smtp.naver.com', 465)
        naver_server.login(admin_email, admin_passwd)
        naver_server.sendmail(admin_email, user_email, msg.as_string())

        naver_server.quit()
        logger.debug(f'to {user_email}')


    def _mysql_dml(self, sql, val=None):
        # INSERT, UPDATE, DELETE
        #print("SQL: ", sql)
        #print("VAL: ", val)
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
            self.logger.error("MYSQL: " + set(err))
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
            self.logger.error("MYSQL: " + str(err))
        finally:
            try:
                conn.close()
            except:
                pass
        return rows

    def bulk_insert_db(self, table_name, insert_list):
        value_list = []
        for item in insert_list:
            value_list.append(str(item))
        value_str = ",".join(value_list)
        sql = "INSERT INTO %s (DATE, CLASS, CORR) VALUES "  % table_name
        sql = sql + value_str
        self._mysql_dml(sql)

    def delete_outdated_data(self):
        base_time = self.now_date - datetime.timedelta(seconds=self.max_db_date)
        base_time_str = base_time.strftime("%m%d%H%M%S")
        sql = "DELETE FROM %s WHERE DATE <= " % self.mysql_table
        self.logger.debug("delete outdated data")
        self._mysql_dml(sql + " %s", (int(base_time_str)))

    def run(self):
        self.run_video()
        self.logger.debug("** activate video server")
        while True:
            if self.frame is not None:
                print("** video server is started")
                break
            else:
                time.sleep(1)
        print('** finish')

class ObjectDetector():
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
    NUM_CLASSES = 90

    def download_model(self, model_name):
        model_file = model_name + '.tar.gz'
        print("downloading model", model_name, "...")
        opener = urllib.request.URLopener()
        opener.retrieve(self.DOWNLOAD_BASE + model_file, model_file)
        print("download completed");
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if self.GRAPH_FILE_NAME in file_name:
                tar_file.extract(file, os.getcwd())
                print(self.graph_file, "is extracted");

    def __init__(self, model_name, label_file='data/mscoco_label_map.pbtxt'):
        # Initialize some variables
        print("ObjectDetector('%s', '%s')" % (model_name, label_file))
        self.process_this_frame = True

        # download model
        self.graph_file = model_name + '/' + self.GRAPH_FILE_NAME
        if not os.path.isfile(self.graph_file):
            self.download_model(model_name)

        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            graph = self.detection_graph

            ops = graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = graph.get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, 480, 640)
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            self.tensor_dict = tensor_dict

        self.sess = tf.Session(graph=self.detection_graph)

        # Loading label map
        # Label maps map indices to category names,
        # so that when our convolution network predicts `5`,
        # we know that this corresponds to `airplane`.
        # Here we use internal utility functions,
        # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(label_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.output_dict = None

        self.last_inference_time = 0

    def run_inference(self, image_np):
        sess = self.sess
        graph = self.detection_graph
        with graph.as_default():
            image_tensor = graph.get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(self.tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def time_to_run_inference(self):
        unixtime = int(time.time())
        if self.last_inference_time != unixtime:
            self.last_inference_time = unixtime
            return True
        return False

    def detect_objects(self, frame):
        time1 = time.time()
        # Grab a single frame of video

        # Resize frame of video to 1/4 size for faster face recognition processing
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = frame

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        time2 = time.time()

        # Only process every other frame of video to save time
        if self.time_to_run_inference():
            self.output_dict = self.run_inference(rgb_small_frame)

        time3 = time.time()

        vis_util.visualize_boxes_and_labels_on_image_array(
          frame,
          self.output_dict['detection_boxes'],
          self.output_dict['detection_classes'],
          self.output_dict['detection_scores'],
          self.category_index,
          instance_masks=self.output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=3)

        time4 = time.time()

        #print("%0.3f, %0.3f, %0.3f sec" % (time2 - time1, time3 - time2, time4 - time3))

        return frame
    
    def detect_objects_live(self, frame):
        time1 = time.time()
        # Grab a single frame of video

        # Resize frame of video to 1/4 size for faster face recognition processing
        #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = frame

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        time2 = time.time()

        # Only process every other frame of video to save time
        if self.time_to_run_inference():
            self.output_dict = self.run_inference(rgb_small_frame)

        time3 = time.time()

        _, rtn_dict = vis_util.visualize_boxes_and_labels_on_image_array_live(
          frame,
          self.output_dict['detection_boxes'],
          self.output_dict['detection_classes'],
          self.output_dict['detection_scores'],
          self.category_index,
          instance_masks=self.output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=3)

        time4 = time.time()

        #print("%0.3f, %0.3f, %0.3f sec" % (time2 - time1, time3 - time2, time4 - time3))

        return frame, rtn_dict

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

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


if __name__ == '__main__':
    vr = VideoRun()
    vr.run()
