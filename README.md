# discrete-jit-structured-denoising

이 저장소는 **한 가지 학습 실험만** 남긴 최소 구성입니다.

- 입력: 균일 토큰 corruption 이 들어간 시퀀스(학습은 부분 corruption, 평가는 완전 corruption)
- 출력: clean 시퀀스
- 목표: Transformer 가 corruption 을 직접 복원하도록 학습

현재 기본 설정은 repeating/mirrored/interleaved/arithmetic 패턴을 섞어 더 복잡한 구조를 학습합니다.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --config configs/motif_clean_prediction.yaml --output-dir outputs/motif_clean
```

## Evaluate

```bash
python evaluate.py --config configs/motif_clean_prediction.yaml --checkpoint outputs/motif_clean/best.pt
```

## Sample

```bash
python sample.py --config configs/motif_clean_prediction.yaml --checkpoint outputs/motif_clean/best.pt
```
