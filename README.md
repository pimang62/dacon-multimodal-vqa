# 월간 데이콘 이미지 기반 질의 응답 AI 경진대회

## 1. Introduction

## 2. Data

## 3. Setup

### LLaVA clone
```python
!git clone https://github.com/haotian-liu/LLaVA.git
%cd /content/LLaVA
```

### Install
```python
!pip install --upgrade pip
!pip install -e .
!pip install ninja
!pip install flash-attn --no-build-isolation
```

### Vicuna clone
```python
!git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
```

### Download Data
```python
# Download directly
!gdown https://drive.google.com/u/0/uc?id=1a9XB3r83ZCFWLOHBp8ooz3zQFl9rEIei&export=download
```

### Preprocessing
  * You could get 'output.json' file
```python
# unzip
import zipfile
import os


zip_file_path = '/content/LLaVA/open.zip'

extracted_folder = '/content/'

def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
try:
    extract_zip(zip_file_path, extracted_folder)
    print(f"압축 파일을 성공적으로 해제하였습니다. 경로: {extracted_folder}")
except Exception as e:
    print(f"압축 파일 해제 중 오류가 발생하였습니다: {e}")



# make 'output.json'
import csv
import json

with open('/content/data/train.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)

json_data = []
for row in data:
    id, image_id, question, answer = row
    json_data.append({
        "id": id,
        "image": "/content/data/image/train/" + image_id + ".jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + question
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    })

with open('output.json', 'w') as f:
    json.dump(json_data, f, indent=4)
```

## 4. Train
*

## 5. Re-training


## 6. Inference

