# 공인 IP SSH: 서버 연산 + 노트북 실시간 화면·맵 저장

노트북에서 서버로 SSH 접속할 수 있으면 됩니다. 노트북의 Tailscale, SSH 데몬, 공유기
포트 개방은 필요하지 않습니다. 아래 `SERVER_PUBLIC_IP`와 SSH 포트는 실제 값으로 바꾸세요.

- 노트북: USB RGB-D 캡처, 브라우저 3D 렌더링, 전체 맵 파일 저장.
- 서버: GPU 0에서 MASt3R 추론·추적·평면/깊이 보정. GPU 1의 기존 작업은 건드리지 않습니다.
- 화면: 서버의 Viser가 요약 점군/RGB/자세를 전송하고 노트북 브라우저가 그립니다. 서버에
  DISPLAY/X11/OpenGL 창이 필요하지 않습니다. 기존 데스크톱 뷰어는 기본 실행에서 유지됩니다.
- 저장: 노트북 폴더를 서버에 SSHFS로 연결합니다. 최근 키프레임은 서버 RAM/VRAM에 있고,
  전체 키프레임·맵 청크·보정 자세·성능 로그는 노트북 디스크에 기록합니다. 모델과 코드,
  일반 실행 환경 캐시는 서버에 있어야 합니다. 브라우저 저장 기능을 사용하는 것은 아닙니다.

## 1. 코드 업데이트와 준비 (각 PC에서 한 번)

서버:

```bash
cd ~/cvr-gh/MASt3R-PlanarSLAM
git pull --ff-only
conda activate mast3r-slam
python -m pip install -r requirements-web.txt
sudo apt install sshfs
```

노트북:

```bash
cd ~/GH/MASt3R-PlanarSLAM
git pull --ff-only
```

노트북에 `/usr/lib/openssh/sftp-server`가 없다면 `sudo apt install openssh-sftp-server`로
설치합니다. SSHFS는 서버에서 일반 사용자로 실행되며 `/dev/fuse` 접근이 필요합니다.
기존 CUDA/PyTorch를 재설치할 필요는 없습니다.

## 2. 노트북 터미널 1 — 카메라 송신

```bash
cd ~/GH/MASt3R-PlanarSLAM
conda activate mast3r-slam
python send_rgbd.py --bind 127.0.0.1 --port 9877
```

이미 송신기가 실행 중이면 그대로 사용합니다. SDK가 D455로 표시하는 장치도 이 경로를 사용합니다.

## 3. 노트북 터미널 2 — 저장 폴더 연결

```bash
cd ~/GH/MASt3R-PlanarSLAM
python3 mount_laptop_maps.py \
  --server cvr@SERVER_PUBLIC_IP \
  --local-dir "$HOME/GH/MASt3R-PlanarSLAM/maps/server" \
  --server-mount '~/laptop-maps'
```

SSH 포트가 22가 아니면 `--ssh-port 포트번호`를 추가합니다. 필요하면 `--identity 키파일`을
지정하거나 `--server`에 SSH config 별칭을 사용합니다. SSH 비밀번호 입력이 가능합니다.
마운트 경로에 파일이나 기존 마운트가 있으면 덮어쓰지 않고 중단합니다.
**이 터미널은 SLAM을 종료하고 저장 완료를 확인할 때까지 유지합니다.**
정상 연결 시 `ARCHIVE READY: laptop read/write verified`가 출력됩니다.
`mountpoint`도 멈추는 경우 기존 SSHFS 마운트가 응답하지 않을 수 있습니다. 서버에서
SLAM을 먼저 중지하고 `fusermount3 -u -z "$HOME/laptop-maps"`로 기존 마운트를 해제한 뒤
노트북의 연결 프로그램을 다시 실행합니다. 연결 프로그램은 기존 마운트 여부를
`/proc/self/mountinfo`로 검사하므로 끊긴 마운트에 `stat()` 요청을 보내며 무한 대기하지 않습니다.

구현은 OpenSSH SFTP와 [SSHFS의 passive 모드](https://github.com/libfuse/sshfs/blob/master/sshfs.rst)를
사용합니다. 노트북에서 시작한 SSH 연결을 거꾸로 파일 연결에 사용하므로 서버가 노트북에
새 SSH 접속을 할 필요가 없습니다. 자신이 관리하는 신뢰하는 서버에 연결하세요.

## 4. 노트북 터미널 3 — 카메라·화면 터널과 서버 로그인

기존 `-R`만 있는 연결은 이 명령으로 교체합니다. 이미 같은 포트를 쓰는 터널이 있으면
중복으로 만들지 않습니다. `SERVER_PUBLIC_IP`는 실제 서버 주소로 바꿉니다.

```bash
ssh -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:19877:127.0.0.1:9877 \
  -L 127.0.0.1:8080:127.0.0.1:8080 \
  cvr@SERVER_PUBLIC_IP
```

비표준 SSH 포트는 `ssh -p 포트번호 ...`로 지정합니다. 뷰어는 서버의 127.0.0.1에만
열리고 SSH로 전달됩니다. 웹 포트를 외부에 공개하거나 `DISPLAY`를 지정하지 않습니다.

로그인한 서버에서:

```bash
cd ~/cvr-gh/MASt3R-PlanarSLAM
conda activate mast3r-slam
CUDA_VISIBLE_DEVICES=0 python run_d457.py \
  --camera-host 127.0.0.1 \
  --camera-port 19877 \
  --image-size 640 \
  --fps 5 \
  --web-viz \
  --web-port 8080 \
  --archive-mount "$HOME/laptop-maps"
```

`--archive-mount`는 SSHFS 마운트와 노트북 표식을 확인합니다. 연결되지 않았다면 서버의
일반 폴더로 저장을 대체하지 않고 실패합니다. `--archive-dir`와 동시에 지정하지 않습니다.
서버에 계속 저장하고 화면만 볼 경우 `--archive-mount`만 생략합니다.

## 5. 노트북 브라우저에서 보기

**http://127.0.0.1:8080** 을 엽니다. 카메라 RGB, 요약 점군, 현재 프레임,
현재 카메라 자세, 키프레임 경로, 추적 상태가 표시됩니다.

- 드래그: 회전, 오른쪽 드래그: 이동, 휠: 확대/축소.
- `Reset view`: 초기 시점. `Map`/`Current frame`: 표시 전환, 점 크기 조절 가능.
- 브라우저 종료·새로고침은 SLAM을 중단하지 않습니다. 재접속하면 최신 화면을 받습니다.
- 기본값은 지도 50,000점 + 현재 프레임 최대 4,000점, 최대 2회/초입니다.
  `--web-points 20000 --web-fps 1`로 화면 전송량을 더 줄일 수 있습니다.
- 지도는 최근 키프레임과 메모리의 과거 요약 점군만 사용합니다. 전체 저장 맵을 매번
  읽지 않습니다. 브라우저 미접속 중에는 표시용 GPU→CPU 복사도 생략합니다.
- 좌표에는 기존 Sim3 회전·이동·스케일을 모두 적용합니다. 초기 카메라의 위쪽은 -Y입니다.
  키프레임 경로는 최대 4,096개 자세를 표본화하며 전체 프레임 궤적 표시와는 다릅니다.
- `--web-viz`는 데스크톱 창을 대체합니다. `--no-viz`만 지정하면 화면이 없습니다.
  둘을 함께 지정해도 브라우저는 켜집니다. 노트북에서 렌더링 가능한 WebGL 브라우저가 필요합니다.

기본 50,000점은 좌표 float32 + RGB uint8로 약 0.75 MB이며, 지도 갱신 시 전송됩니다.
현재 점군·이미지·프로토콜 부가 데이터도 있습니다. 카메라 업로드와 **전체 맵 파일 저장**은
이 화면 전송량과 별개입니다. 실제 인터넷 연결에서 FPS와 지연은 측정해야 합니다.

## 6. 종료와 저장 맵 열기

1. 서버 SLAM 터미널에서 Ctrl+C 후 `Saved map:`과 `Summary:`가 나올 때까지 기다립니다.
2. 서버에서 `fusermount3 -u "$HOME/laptop-maps"`로 저장 연결을 해제합니다.
3. 노트북 저장 연결/카메라 프로그램을 종료합니다. SSH 터널도 종료해도 됩니다.

맵은 **노트북의 `~/GH/MASt3R-PlanarSLAM/maps/server/<세션시각>/`**에 있습니다.
기존 저장 맵 뷰어로 전체 상세 데이터를 볼 수 있습니다.

```bash
cd ~/GH/MASt3R-PlanarSLAM
conda activate mast3r-slam
./view_saved_map.sh maps/server/세션시각
```

전체 키프레임 텐서·RGB·깊이·특징·점군과 비키프레임 요약 저장 정책을 유지합니다.
연속 영상 녹화는 아닙니다. 연결이 끊기면 저장/재로딩이 실패할 수 있으며, 완료된 파일은
노트북에 남지만 진행 중이던 세션의 완전성을 보장하지 않습니다. 연결 단절 후 기존 세션에
자동으로 이어 쓰지 않습니다. SLAM 종료 후 남은 마운트를 해제하고 새 연결/세션을 시작합니다.

## 성능과 검증 범위

노트북 저장은 종료 후 복사가 아니라 실행 중 SSHFS 쓰기입니다. 서버 디스크에 전체 맵
복제본을 만들지 않습니다. 원격 저장은 네트워크 왕복과 대역폭이 필요하며, 기존의 제한된
저장 큐가 차면 추적도 기다립니다. 과거 키프레임 재검색 역시 노트북에서 읽어 오므로 느려질
수 있습니다. **저장 손실 없이 실시간 FPS도 항상 유지한다고 보장하지 않습니다.**
`performance.jsonl`의 `step_ms`, `map_wait_ms`, `stages_ms.keyframe_archive`를 확인하세요.

로컬 검증: CPU 점군/좌표/마운트 방어 테스트, 합성 CUDA 추적 16프레임의 브라우저 경로,
실제 Chrome의 점군·RGB 표시와 재접속, 실제 SSHFS/SFTP의 파일 쓰기·rename·재읽기를 확인했습니다.
SSHFS 경로로 합성 16프레임을 처리해 키프레임 6개·비키프레임 10개를 노트북 쪽 폴더에
저장하고 기존 맵 리더로 읽었습니다. 마운트 해제 후 서버 쪽 폴더에 맵이 남지 않는 것도 확인했습니다.
공인 IP 회선/A6000 서버에서의 왕복 지연·처리율은 별도로 확인해야 합니다.

[Viser 공식 API](https://viser.studio/main/api/core/server/)를 사용하는 선택적 뷰어이며,
`requirements-web.txt`로 버전을 고정합니다. 기존 저장 데이터 형식은 바꾸지 않습니다.
