# MASt3R-PlanarSLAM

[MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM)을 기반으로 RealSense RGB-D, 평면 제약, 깊이 보정, 디스크 맵 저장과 원격 카메라 입력을 추가한 실험 프로젝트입니다. 기존 MASt3R 모델·추적·대응점 검색·시각화 구조를 사용합니다.

## 구현 기능

- RealSense D4xx USB RGB-D: 컬러/깊이 정렬, 렌즈 보정, 미터 단위 깊이.
- 실측 깊이 우선 융합 및 모델 깊이 대조·보완. 큰 평면과 반복 관측 표면의 제한적인 시간축 깊이 보정.
- 평면 제약과 승인된 RGB-D 대응에 기반한 비동기 전역 자세 그래프.
- 최근 8키프레임 + 과거 후보 캐시, 디스크 키프레임 보관, 대응점 선별과 단계별 성능 측정.
- 전체 키프레임 정보 및 비키프레임 요약 점군 저장. 실시간 요약 뷰어와 저장 맵 상세 뷰어 분리.
- 카메라 PC → GPU 서버 RGB-D 전송: RGB JPEG/PNG, 무손실 깊이, 보정값·촬영 시각, 최신 프레임 요청 방식.
- SSH 접속 서버의 브라우저 실시간 맵(`--web-viz`)과 노트북 디스크에 직접 맵 저장(`--archive-mount`).
- 512×384 / 640×480 처리 경로와 선택적 오프라인 맵 외관·메시 실험 도구.

평면·보정 알고리즘은 실험적 확장입니다. 장거리 궤적 정확도나 자율주행에 필요한 지도 정확도를 검증한 제품은 아닙니다. 640 처리는 512 학습 체크포인트의 추론 해상도 확장이며 재학습 모델이 아닙니다.

## 문서

- [기능·설정·실사 검증 기록](README_PLANAR.md)
- [카메라 PC / GPU 서버 네트워크 실행](README_NETWORK_RGBD.md)
- [공인 IP SSH: 서버 연산 + 노트북 화면·실시간 저장](README_REMOTE_LIVE.md)
- [원본 프로젝트 안내](README_UPSTREAM.md)
- [원본 출처와 포함된 의존성](THIRD_PARTY.md)

## 설치

Linux, NVIDIA GPU 및 CUDA 개발 도구가 필요합니다. 이 저장소는 의존성 소스 스냅샷을 포함하므로 `git submodule update`가 필요하지 않습니다. 모델 가중치·가상환경·컴파일된 CUDA 라이브러리·촬영 데이터·생성 맵은 포함하지 않습니다.

```bash
git clone https://github.com/PGH-snfl/MASt3R-PlanarSLAM.git
cd MASt3R-PlanarSLAM
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
```

먼저 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라 GPU 및 CUDA Toolkit에 맞는 `torch`, `torchvision`을 설치합니다. CUDA 확장을 빌드하므로 `nvcc --version`과 `python -c "import torch; print(torch.__version__, torch.version.cuda)"`도 확인합니다.

그 다음 프로젝트 의존성을 설치합니다.

```bash
python -m pip install setuptools wheel ninja 'numpy==1.26.4' faiss-cpu
python -m pip install --no-build-isolation -e thirdparty/mast3r
python -m pip install --no-build-isolation -e thirdparty/in3d
python -m pip install --no-build-isolation -e .
python scripts/download_checkpoints.py
```

모든 CUDA 확장은 실행할 GPU 환경에서 새로 빌드합니다. 다른 PC의 `.so`나 Conda 환경을 그대로 복사하지 않습니다. 위는 소스 설치 순서이며 새 서버에서의 완전한 재설치 검증은 아직 수행하지 않았습니다.

개발 PC에서 확인한 주요 버전은 Python 3.11, torch 2.13.0+cu132, torchvision 0.28.0+cu132, NumPy 1.26.4, pyrealsense2 2.58.4.10922, OpenCV 5.0.0.93입니다. 이는 개발 환경 기록이며 모든 서버에 동일 버전을 강제하는 잠금 파일이 아닙니다.

## USB 카메라 실행

```bash
conda activate mast3r-slam
python run_d457.py --list-devices
python run_d457.py --check-camera
python run_d457.py
```

기본 처리 크기는 640×480, 처리율 상한은 5 FPS입니다. `--image-size 512`로 512×384를 선택하고, 화면 없이 실행할 때는 `--no-viz`를 추가합니다. `./run_d457.sh`도 사용할 수 있습니다. 실행 스크립트는 활성 Conda 환경 또는 `MAST3R_PYTHON` 경로를 사용합니다.

## A6000 등 다른 서버로 카메라 전송

카메라가 연결된 PC:

```bash
python send_rgbd.py --bind "$(tailscale ip -4)"
```

GPU 서버 (`CAMERA_PC_IP`는 카메라 PC의 Tailscale IP로 교체):

```bash
python check_rgbd.py --host CAMERA_PC_IP
python run_d457.py --camera-host CAMERA_PC_IP --image-size 640 --fps 5
```

카메라 송신은 CPU만 사용하며 녹화 파일을 만들지 않습니다. 기본 맵 저장 위치는 서버입니다.
SSH 터미널에서는 `--web-viz`로 노트북 브라우저에 실시간 화면을 표시할 수 있습니다.
`mount_laptop_maps.py`와 `--archive-mount`를 함께 사용하면 전체 맵 파일도 실시간으로
노트북 디스크에 저장합니다. [공인 IP SSH 실행 순서](README_REMOTE_LIVE.md)를 참고하세요.
화면이 전혀 필요 없으면 `--no-viz`를 사용합니다.

## 저장 맵

```bash
python view_saved_map.py --help
./view_saved_map.sh maps/SESSION
```

실행 중 생성한 맵은 `maps/<시각>/`에 저장됩니다. 키프레임의 RGB·깊이·특징·점군·자세는 보관하고 비키프레임은 요약 점군만 보관합니다. 이 데이터는 Git에 올리지 않습니다. 오프라인 외관/메시 스크립트는 선택적 실험이며 별도 의존성이 필요합니다. [상세 문서](README_PLANAR.md)를 참고하세요.

## 검증 범위

```bash
python tests/test_rgbd_transport.py -v
python -m unittest discover -s tests -p 'test_*.py'
```

일부 테스트에는 설치된 CUDA 확장과 GPU가 필요합니다. 실사/벤치마크 스크립트는 `tests/`에 있습니다. 저장소에 생성 보고서와 촬영 맵은 포함하지 않으며 기존 개발 PC의 검증 기록은 상세 문서에 요약했습니다.

네트워크 입력은 실제 USB RealSense → 로컬 TCP → MASt3R 모델 추론 → 맵 저장까지 8프레임 검증했습니다. 다른 네트워크의 A6000 서버 성능과 장거리 지도 정확도는 아직 검증하지 않았습니다.

## 라이선스

원본 MASt3R-SLAM의 **CC BY-NC-SA 4.0**을 유지합니다. [LICENSE.md](LICENSE.md)와 [THIRD_PARTY.md](THIRD_PARTY.md)를 확인하세요. 포함된 의존성과 모델 가중치에는 각각의 라이선스가 적용됩니다.
