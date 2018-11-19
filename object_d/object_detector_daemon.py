# object_detector.py

import cv2
import os
import numpy as np
import tensorflow as tf
import tarfile
import six.moves.urllib as urllib
import time
import face_recog
import threading
import socket
import pickle
import struct
import Socket
import datetime

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops


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

class VideoRun():
    def __init__(self):
        self.frame = None
        self.frame_cnt = 0
        self.video_server_host = '0.0.0.0'
        self.video_server_port = 5051

        self.current_time = 3
        self.current_buffer = []

    def run_video(self):
        print("run video")
        import camera
        face_recog_m = face_recog.FaceRecog()
        detector = ObjectDetector('ssd_mobilenet_v1_coco_2017_11_17')
        print(face_recog_m.known_face_names)

        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.

        def func(face_recog_m, detector):
            print("start func")
            retry_cnt = 0
            while True:
                now_date = datetime.datetime.now()
                now_str = now_date.strftime("%Y%m%d%H%M%S")
                try:
                    frame, face_result_list = face_recog_m.get_frame_live()
                    frame, obj_detection_dict = detector.detect_objects_live(frame)
                    for face_result in face_result_list:
                        face_corr = face_result[0]
                        face_name = face_result[1]
                        print(face_name)
                        self.current_buffer.append((now_date, face_name))
                    for item in obj_detection_dict:
                        try:
                            obj_corr = item
                            class_str = obj_detection_dict[item][0]
                            class_list = class_str.split()
                            obj_class = class_list[0][:-1]
                            print(obj_class)
                            obj_score = int(class_list[1][:-1])
                            self.current_buffer.append((now_date, obj_class))
                        except:
                            pass
                    origin_len = len(self.current_buffer)
                    for _ in range(origin_len):
                        if self.check_current_max(now_date):
                            del self.current_buffer[0]
                    self.frame = frame
                    self.frame_cnt += 1
                except Exception as err:
                    #print(err)
                    try:
                        face_recog_m = face_recog.FaceRecog()
                    except:
                        retry_cnt += 1
                        if retry_cnt > 5:
                            break

        th = threading.Thread(target=func, args=(face_recog_m, detector))
        th.daemon = True
        th.start()
        return th

    def check_current_max(self, now_date):
        if len(self.current_buffer) > 0:
            last_date_time = self.current_buffer[0][0]
            if now_date - last_date_time >= datetime.timedelta(seconds=self.current_time):
                return True
        return False

    def run_server(self):
        WELCOME_MSG = b"+OK Welcome Video AI Server\r\n"
        while True:
            s = Socket.Socket()
            s.Bind(self.video_server_port)
            sock = s.Accept()
            print("Accept")
            sock.SendMessage(WELCOME_MSG)
            while True:
                try:
                    line = sock.Readline()
                    line = line.decode('utf-8')
                    cmd, param = line.strip().split() 
                    print("CMD : ", cmd)
                    print("param : ", param)
                    ret_message = b"-ERR BAD\r\n"
                    if cmd.upper() == "GET_CURRENT":
                        ret_message = b'+OK good\r\n'
                    elif cmd.upper() == "EXISTS":
                        ret_message = b'+OK i dont  know\r\n'
                        ret_message = self.EXISTS(obj=param)
                    elif cmd.upper() == "SHOW_CURRENT":
                        ret_message = b'+OK i dont  know\r\n'
                        ret_message = self.SHOW_CURRENT()
                    elif cmd == "QUIT":
                        break
                    print("CMD : **", cmd)
                    sock.SendMessage(ret_message)
                except Exception as err:
                    print(err)
                    break
            try:
                sock.close()
                s.close()
            except: pass

    def EXISTS(self, obj=None):
        if obj == None:
            return b"+OK 1"
        else:
            return b"+OK 1"
 
    def SHOW_CURRENT(self):
        tmp_set = set([])
        for obj in self.current_buffer:
            tmp_set.add(obj[1])
        rtn_str = ""
        for item in tmp_set:
            rtn_str += "%s, " % item
        rtn_string = '+OK %s\r\n' % rtn_str
        b = bytes(rtn_string, 'utf-8')
        return b

    def run(self):
        th = self.run_video()
        print("run video AI start")
        while True:
            if self.frame is not None:
                print("server is started")
                break
            else:
                time.sleep(1)

        self.run_server()
        th.join()
        print('finish')


if __name__ == '__main__':
    vr = VideoRun()
    vr.run()
