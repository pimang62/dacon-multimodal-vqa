# 월간 데이콘 이미지 기반 질의 응답 AI 경진대회
<img width="502" alt="image" src="https://github.com/pimang62/dacon-multimodal-vqa/assets/121668884/ed1b318f-f6c6-41db-adba-504f16e15b64">

## 1. Introduction

## 2. Data

## 3. Setup

### LLaVA clone
```python
!git clone https://github.com/haotian-liu/LLaVA.git
%cd /content/LLaVA
```

### Install
```
!pip install --upgrade pip
!pip install -e .
!pip install ninja
!pip install flash-attn --no-build-isolation
```

### Vicuna clone
```
!git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
```

### Download Data
```
# Download directly
!gdown https://drive.google.com/u/0/uc?id=1a9XB3r83ZCFWLOHBp8ooz3zQFl9rEIei&export=download
```

### Pre-processing
  * You could get output.json file
```
!python preprocessing.py
```

## 4. Train
*

## 5. Re-training


## 6. Inference

