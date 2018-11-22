# nugu_project : SAURON (2018.11)

- 영상 데이터를 딥러닝, 패턴 인식 등의 open source를 활용하여, 영상 처리하여 유의미한 요약 데이터를 추출
  - 얼굴 인식, 얼굴 구별, 객체인식, 객체의 위치 좌표, 움직임
- 추출된 데이터 디비 서버에 업로드
- proxy server를 통해 디비 서버와 인터페이스간 REST API 통신
- 인공지능 스피커(NUGU)와 PROXY 서버간의 통신을 통한 음성 인터페이스 제공
  - https://developers.nugu.co.kr/#/

### 개발 환경
- python 3.6 anaconda3
- flask
- tensorflow
- opencv
- mysql 5.7
- window, mac os

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

### 서버 구성

#### 지능형 영상 처리 서버 (video AI server)
- 사용자단에서 구축하는 원격지(물리적으로 격리된 감시공간)의 서버
- 카메라 입력 소스와 같은 내부망으로 구성
  - 원본 영상 데이터의 보안
- 카메라에서 입력되는 영상 프레임을 딥러닝으로 분석하여 결과를 디비 서버로 전송 (mysql)
- 주기적으로 (약 10분) 프레임 캡쳐 이미지를 서버내 보관 
  - 요청시 해당 시간의 이미지를 등록된 사용자의 메일로 전송

#### mysql db
- 카메라에서 추출된 데이터를 정형 데이터베이스에 적재
- 설정값에 정한 시간만큼의 데이터를 저장
- 영상처리 서버와 프록시 서버간의 통신 매개

#### NUGU 인터페이스 서버 (interface proxy)
- 사용자가 NUGU에 음성 명령을 내리면 proxy server로 REST API 통신
- 입력된 파라메터에 따른 액션 수행
- 사용자 발화의 의도(INTENT) 에 맞는 SQL수행
- 영상 요약 데이터를 취합 분석하여 그에 맞는 결과 출력
  - 1시간전에 수상한 사람 있었어? => 아니요 없었어요.
  - 지금 철수 있니? => 네 멍멍이랑 함께 있네요
  - 지금 영희 있니? => 30분전에 떠났어요
  - 지금 A창고 상황 => c,b,a 가 있네요
  
  
### 실행 방법

- 프로젝트 폴더로 이동

```
$ cd  object_d
```

- 사람 얼굴 등록
  - knowns 폴더에 얼굴이 포함된 사진 (.jpg) 등록
  - 예시 ) chulsu.jpg


- conf 파일 수정

```
$ vi server_conf.py

// 사람 얼굴 사진을 등록한 경우 conf 파일의 dictionary 에 추가
// 이미지 파일명을 키로 음성인식으로 불릴수있는 값을 리스트에 입력
// 예시 ) "chulsu" : ["철수", "김철수", "철수씨"] 

// 카메라 입력 소스 수정
```

- 라이브 스트리밍 서버를 띄울 경우 camera.py에서 카메라 입력 소스 변경 (웹캠 또는 파일경로(mp4))

```
$ vi camera.py
```

- 백그라운드에서 이미지 영상 처리하는 서버 실행

```
$ python background_video_server_v2.py
```

- NUGU와 통신하는 인터페이스 프록시 서버 실행 (5060 포트)

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

## licence
- 무단으로 상업적 사용 불가
- 문의 : beobest2@gmail.com
