# RealSense RGB-D를 다른 GPU 서버에서 처리하기

카메라가 연결된 PC는 RGB·깊이를 수집하고, A6000 서버는 기존 MASt3R-PlanarSLAM
추론·깊이 보정·평면 처리를 수행합니다. 아래는 사설망/Tailscale에서의 카메라 입력 설정입니다.
**노트북에 Tailscale 없이 공인 IP SSH를 사용하고, 실시간 화면과 맵 저장을 노트북에서
진행하려면 [서버 연산 + 노트북 화면·저장 안내](README_REMOTE_LIVE.md)를 따르세요.**

## 1. 카메라 PC에서 송신기 실행

다른 RealSense 실행 프로그램을 종료한 뒤 실행합니다.

```bash
cd ~/GH/MASt3R-PlanarSLAM
conda activate mast3r-slam
tailscale ip -4
python send_rgbd.py --bind "$(tailscale ip -4)"
```

`--bind`는 **카메라가 연결된 이 PC의 Tailscale IP**입니다. 서버 IP를 넣지 않습니다.
송신기는 이 주소의 TCP 9877 포트에서 서버의 요청을 기다립니다. Tailscale 정책과
호스트 방화벽에서 서버 → 카메라 PC의 해당 포트 통신이 허용돼 있어야 합니다.
Tailscale 외부 공개 포트로 열지 않고, 신뢰하는 사설 연결에서 사용합니다.
프로토콜 자체에는 별도 암호화/인증이 없으며 `pickle`은 사용하지 않습니다.
기본 주소는 테스트용 `127.0.0.1`이므로 원격 사용 시 `--bind` 지정이 필요합니다.

송신기에는 모델, PyTorch, CUDA가 필요 없습니다. Python의 `numpy`, `opencv-python`,
`pyrealsense2`만 필요하며 현재 `mast3r-slam` 환경에서 실행 가능합니다.
`./send_rgbd.sh`도 사용할 수 있고 `RGBD_PYTHON`으로 Python 경로를 지정할 수 있습니다.

기본값은 RGB/깊이 640×480, 카메라 30 FPS, 송신 상한 10 FPS입니다.
여러 카메라가 있으면 `python send_rgbd.py --list-devices`로 확인한 후 `--serial`을 지정합니다.

## 2. 서버에서 수신 먼저 확인

현재 수정된 프로젝트 코드를 서버에도 복사해야 합니다. `<카메라_PC_IP>`는
1단계에서 확인한 주소로 바꾸고, 프로젝트 위치가 다르면 `cd` 경로도 바꿉니다.

```bash
cd ~/GH/MASt3R-PlanarSLAM
conda activate mast3r-slam
python check_rgbd.py --host <카메라_PC_IP> --frames 30 --fps 5
```

이 검사는 모델을 로딩하거나 영상을 저장하지 않습니다. 해상도, 보정값 K, 실측 깊이,
프레임 크기, 요청 시간을 출력합니다. 카메라 PC 송신기는 검사 종료 후 계속 켜둡니다.
수신 검사에도 `numpy`, `opencv-python`만 필요합니다.

## 3. 서버에서 SLAM 실행

서버에 모델 체크포인트와 MASt3R 실행 환경을 먼저 준비하고, CUDA 확장을 해당 서버의
GPU에 맞게 빌드합니다. 로컬 Conda 환경이나 컴파일된 `.so`를 그대로 복사해 쓰지 않습니다.

```bash
python run_d457.py --camera-host <카메라_PC_IP> --image-size 640 --fps 5
```

SSH 터미널에서 화면이 필요 없다면:

```bash
python run_d457.py --camera-host <카메라_PC_IP> --image-size 640 --fps 5 --no-viz
```

`--no-viz`는 화면만 끄고 기존 맵 저장을 계속합니다. 맵은 **서버 프로젝트의 `maps/`**에
저장됩니다. `--archive-dir`로 새 세션 디렉터리를 지정할 수 있습니다.
브라우저 화면은 `--web-viz`, 노트북 저장은 `--archive-mount`를 사용합니다.
모델 없이 SLAM 입력 형상까지 확인하려면 `--check-camera`를 추가합니다.
연결 정보는 `--camera-port 9877`, `--camera-timeout 10`으로 조절합니다.
`--camera-host`를 생략하면 기존 USB 직접 입력 방식으로 실행됩니다.

## 전송 방식과 지연

- SDK에서 동기화한 RGB/깊이 한 쌍을 사용하고, 깊이를 컬러 좌표계로 정렬합니다.
- 기존 USB 입력과 같은 SDK 왜곡 보정 LUT로 RGB·깊이를 함께 보정합니다.
- RGB는 기본 JPEG 품질 95입니다. 완전 무손실 RGB는 송신기에 `--rgb-codec png`를 지정합니다.
- 깊이는 정렬/보정된 uint16 값을 zlib로 **무손실** 압축합니다. 깊이 단위(m/unit), K,
  RGB/깊이 촬영 시각, 프레임 번호, 세션 ID를 같은 패킷에 보냅니다.
- 서버에서 깊이를 미터 단위로 바꾸고 기존 유효 거리 필터와 512/640 전처리를 적용합니다.
  실측 깊이가 없는 구간에 대한 모델 깊이 대체 및 기존 보정은 기존 설정을 따릅니다.
- 카메라는 계속 캡처하지만 메모리에는 최신 한 쌍만 보관합니다. 서버가 요청할 때만
  최신 프레임을 압축·전송합니다. 모델 로딩, 일시 정지, 느린 추론 동안 과거 영상이 쌓이지 않습니다.
- 전송 중인 한 프레임에는 실제 네트워크 지연이 발생합니다. 지연 제거 기능은 아니며
  프레임당 왕복 요청도 포함됩니다. 낮은 FPS/큰 이동 간격에서 추적 품질은 별도 검증해야 합니다.
- 단일 SLAM 클라이언트용입니다. 검사 프로그램과 SLAM을 동시에 접속시키지 않습니다.
- 송신기는 영상 파일을 기록하지 않습니다. SLAM의 기존 키프레임/맵 저장은 유지됩니다.
- 연결 끊김·패킷 수신 제한시간 초과·촬영 시각 역전·보정값 변경 시 SLAM을 중단하고
  기존 종료 경로로 맵 쓰기를 정리합니다. 자동 재접속으로 기존 맵을 이어 붙이지 않습니다.
  연결을 복구한 뒤 새 SLAM 세션으로 다시 실행합니다. 서버가 종료되면 송신기는 다음 연결을 기다립니다.

`check_rgbd.py`의 `request`는 프레임 대기·압축·네트워크·복호화를 합친 시간입니다.
`sender_age`는 송신 PC 최신 프레임 슬롯의 대기 시간으로 **네트워크 지연을 포함하지 않습니다**.
양쪽 PC 시계 동기화 없이 편도 지연으로 해석하면 안 됩니다.
SLAM의 `performance.jsonl`에도 `network` 항목으로 이 수치와 패킷 크기, 건너뛴 프레임 수를 저장합니다.

## 검증

```bash
python tests/test_rgbd_transport.py -v
python -m unittest discover -s tests -p test_rgbd.py -v
```

첫 검사는 패킷 분할 수신, RGB 채널 순서, 깊이 무손실 복원, 카메라 변경, 시간 역전,
연결 단절, 수신 제한시간, 프레임 대기열 억제 및 512/640에서 깊이·K 일치를 검증합니다.
실제 다른 서버/Tailscale 회선의 지연과 A6000 처리량은 서버 연결 후 별도 측정해야 합니다.

2026-10-13 로컬 검증 결과:

- 전송/입력 테스트 10개와 기존 USB RGB-D 테스트 7개 통과.
- SDK 표시 `RealSense D455` 장치로 640×480 실제 프레임 15개 전송 확인.
  JPEG 품질 95에서 평균 220.3 KiB/프레임, 로컬 TCP 요청 시간 중앙값 6.9 ms.
  장면·압축 방식에 따라 크기가 달라지며, 이 값은 인터넷/Tailscale 성능이 아닙니다.
- 실제 모델로 네트워크 입력 8프레임을 처리해 초기화 후 7프레임 추적,
  키프레임 1개와 비키프레임 7개 맵 저장 확인. 모델 깊이 보완 및 시간축 표면 보정 동작.
  검증 맵은 임시 폴더에서 생성 후 삭제했고 수치/실행 로그만
  `reports/network_rgbd_20261013/`에 보관했습니다. 이 검사는 `--no-viz`로 수행했습니다.
