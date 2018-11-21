label_dict = {
"person" : ["사람", "인간"],
"bicycle" : ["자전거", "따릉이"],
"car" : ["자동차", "자가용", "승용차"],
"motorcycle" : ["오토바이"],
"airplane" : ["비행기"],
"bus" : ["버스"],
"train" : ["기차", "열차"],
"truck" : ["트럭"],
"boat" : ["배", "보트"],
"traffic light" : ["신호등"],
"fire hydrant" : ["소화전"],
"stop sign" : ["정지표지판"],
"parking meter" : ["주차계측기"],
"bench" : ["벤치"],
"bird" : ["새"],
"cat" : ["고양이", "야옹이", "냐옹이"],
"dog" : ["개", "강아지", "멍멍이"],
"horse" : ["말"],
"sheep" : ["양"],
"cow" : ["소"],
"elephant" : ["코끼리"],
"bear" : ["곰"],
"zebra" : ["얼룩말"],
"giraffe" : ["기린"],
"backpack" : ["백팩", "가방"],
"umbrella" : ["우산"],
"handbag" : ["핸드백", "손가방"],
"tie" : ["넥타이"],
"suitcase" : ["여행가방"],
"frisbee" : ["프리즈비"],
"skis" : ["스킨"],
"snowboard" : ["스노우보드"],
"sports ball" : ["공"],
"kite" : ["연"],
"baseball bat" : ["야구방망이"],
"baseball glove" : ["야구글러브"],
"skateboard" : ["스케이트보드"],
"surfboard" : ["서핑보드"],
"tennis racket" : ["테니스라켓"],
"bottle" : ["병"],
"wine glass" : ["와인잔"],
"cup" : ["컵"],
"fork" : ["포크"],
"knife" : ["칼"],
"spoon" : ["숟가락"],
"bowl" : ["그릇"],
"banana" : ["바나나"],
"apple" : ["사과"],
"sandwich" : ["샌드위치"],
"orange" : ["오렌지"],
"broccoli" : ["브로콜리"],
"carrot" : ["당근"],
"hot dog" : ["핫도그"],
"pizza" : ["피자"],
"donut" : ["도넛"],
"cake" : ["케이크"],
"chair" : ["의자"],
"couch" : ["쇼파"],
"potted plant" : ["화분"],
"bed" : ["침대"],
"dining table" : ["식탁"],
"toilet" : ["화장실"],
"tv" : ["티비", "티브이", "텔레비젼"],
"laptop" : ["노트북", "맥북"],
"mouse" : ["마우스"],
"remote" : ["리모콘", "리모컨"],
"keyboard" : ["키보드"],
"cell phone" : ["휴대폰", "핸드폰", "스마트폰"],
"microwave" : ["전자렌지"],
"oven" : ["오븐"],
"toaster" : ["토스터기"],
"sink" : ["싱크대"],
"refrigerator" : ["냉장고"],
"book" : ["책"],
"clock" : ["시계"],
"vase" : ["꽃병"],
"scissors" : ["가위"],
"teddy bear" : ["곰돌이", "곰인형", "테디베어"],
"hair drier" : ["드라이기", "드라이어"],
"toothbrush" : ["칫솔"],

#  face 등록 정보 추가
"HYUNWOO" : ["현우", "박현우"],
"HAEJOON" : ["해준", "이해준", "해준이"]
}

# mysql connection info
mysql_host = "61.82.116.184"
mysql_user = "root"
mysql_password = "tkdnfhs12#$"
mysql_db = "testcam01"
mysql_table = "tb"
mysql_img_call_table = "imgcall"
mysql_img_file_table = "imgfile"

create_sql_table = "create table tb (DATE int not null, CLASS char(100), CORR char(100))"
create_sql_imgcall_table = "create table imgcall (DATE_CALL int not null, TIME int)"
create_sql_imgfile_table = "create table imgtable (DATE int not null, FILE_PATH char(100))"

# db 입력 간격(3초)
db_insert_term = 1
# buffer 입력 간격 1초
buffer_write_gap = 1
# db file 저장 시간 최대 범위(2 시간)
max_db_date = 1200

# img file 저장 최대 개수 (1000장)
max_img_file_cnt = 100
# 이미지 파일 저장 경로
imwrite_path = "./imgfile/"
# 이미지 파일 저장 시간 간격(10분 마다)
img_write_gap = 60

# proxy server connection info
proxy_host = '0.0.0.0'
proxy_port = 5060

# 현재 시간이라고 볼수있는 시간 범위 - 최근 레코드 개수
now_time_range = 3
# 과거 특정 시점 질문시 +-10 분 조회 
past_range = 10

# camera input source
camera_source=0
#camera_source="/Users/hwpark/Desktop/test.mp4"
#camera_source="./test.mp4"

user_email=["cleanby@naver.com", "haejoon309@naver.com"]
