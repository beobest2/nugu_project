# nugu_project
- 개발 환경 세팅

```
Install anaconda 3.6

python3 -m venv py3
source py3/bin/activate

pip install --upgrade pip
pip install opencv-python
pip install opencv-contrib-python
pip install dlib
pip install face_recognition
pip install flask
pip install --upgrade tensorflow
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
pip install PyMySQL
```

- 프로젝트 폴더로 이동

```
$ cd  object_d
```

- 카메라 입력 소스 변경 (웹캠 또는 파일경로(mp4))

```
$ vi camera.py
```

- 백그라운드에서 이미지 영상 처리하는 서버 실행 포트

```
$ python background_video_server.py
```

- 위 서버와 통신하여(내부 5051 포트)  NUGU와 통신하는 인터페이스 flask 서버 실행 (5060 포트)

```
$ python nugu_interface_server.py
```

- 영상 뷰어 서버 실행 (5050 포트)

```
$ python live_streaming.py
```

- TEST

```
// health check
127.0.0.1/health

// 최근 10초내의 객체 목록 출력
127.0.0.1:5060/show 

// 영상 스트리밍 (동영상 파일의 경우 스트리밍 서버 접속시부터 실행되므로 NUGU와 싱크가 맞지 않을 수있음)
127.0.0.1:5050
```
