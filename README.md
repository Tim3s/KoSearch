# KoSearch Lab

한국어 검색 알고리즘을 동일한 문서와 qrels에서 비교하는 재현 가능한 검색 실험 프로젝트입니다. BM25, multilingual E5/BGE-M3 dense retrieval, 점수 정규화 hybrid, 검색 품질 지표, 지연 시간·메모리 측정과 FastAPI를 제공합니다. 각 함수의 docstring은 역할과 주요 동작을 한국어 또는 영어로 설명합니다.

## 환경

- 검증 환경: Windows, Conda `KS`, Python 3.14.10, NVIDIA GeForce RTX 3090 (24 GB)
- CUDA PyTorch, Hugging Face `datasets`, Sentence Transformers, FAISS CPU, FastAPI를 `requirements.txt`에 둡니다.
- Dense model weights는 첫 사용 때 Hugging Face에서 다운로드되며 Git에 포함하지 않습니다.

새 환경에서는 PyTorch 공식 CUDA wheel을 먼저 설치한 뒤 프로젝트 의존성을 설치합니다. 이 프로젝트 검증 환경에서는 CUDA 12.8 wheel과 RTX 3090을 사용했습니다.

```powershell
conda activate KS
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

GPU 확인:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 구현

- `app/schemas.py`: 문서·질의·qrels 참조를 검증하는 공통 데이터 타입
- `app/data/autorag.py`: MTEB 형식 corpus/queries/qrels 로더. 기본 데이터는 `mteb/AutoRAGRetrieval` test split입니다.
- `app/retrieval/bm25.py`: 외부 모델이 필요 없는 BM25 Okapi 역색인 기준선
- `app/models/embedder.py`: 배치 임베딩. E5는 query/passage prefix를 붙이고 BGE-M3는 별도 prefix 없이 인코딩합니다.
- `app/retrieval/dense.py`: L2 정규화, cosine 점수, FAISS `IndexFlatIP`, 모델별 저장/복원. FAISS가 없으면 동일한 exact search NumPy backend를 사용합니다.
- `app/retrieval/hybrid.py`: BM25와 Dense 후보 점수를 각각 min-max 정규화한 뒤 `alpha * dense + (1-alpha) * bm25`로 결합합니다.
- `app/evaluation/metrics.py`, `app/services/benchmark.py`: Recall@K, MRR@K, graded nDCG@K, 평균·중앙값·p95 latency, 문서 encoding 시간, 프로세스 RSS와 CUDA 메모리를 측정합니다.
- `app/main.py`, `app/api/`: 문서 등록, 검색, 평가, 실험 조회와 health API
- `scripts/evaluate.py`: JSON·CSV 결과를 저장하는 재현 실험 CLI

BM25 토크나이저는 연속된 한글·영문·숫자 문자열로 나누며 형태소 분석이나 조사 제거는 하지 않습니다. 검색 결과 차이에 영향을 줄 수 있는 제한입니다. 인덱스 재등록은 전체 문서 교체이고 중복 문서 ID는 거부합니다. Hybrid 점수는 각 검색기가 반환한 후보 집합 내 min-max 정규화 결과이며, 상위 `3 * top_k` 후보를 합칩니다.

## 데이터셋

[MTEB AutoRAGRetrieval](https://huggingface.co/datasets/mteb/AutoRAGRetrieval)은 `corpus`, `queries`, `qrels`를 제공하며 페이지에 MIT 라이선스가 표시되어 있습니다. 현재 test split은 720개 문서, 114개 질의, 114개 qrels 질의입니다. 원본 파일은 Hugging Face 캐시에 저장하고 프로젝트에 복사하지 않습니다. `yjoonjang/markers_bm`의 qrels-only 미리보기 대신 corpus와 query 텍스트가 있는 MTEB 저장소를 기본 로더에 사용합니다.

## 평가

BM25 기준선:

```powershell
python scripts/evaluate.py --methods bm25
```

E5 Dense, BGE-M3 Dense, Hybrid alpha 비교:

```powershell
python scripts/evaluate.py --methods dense --model intfloat/multilingual-e5-base --device cuda
python scripts/evaluate.py --methods dense --model BAAI/bge-m3 --device cuda
python scripts/evaluate.py --methods bm25 dense hybrid --model intfloat/multilingual-e5-base --device cuda --alpha 0.25 0.5 0.75
```

모델별 비교 결과를 서로 구분해 저장하려면 `--output-dir experiments/e5` 또는 `--output-dir experiments/bge-m3`를 지정합니다. `--revision`에 Hugging Face commit SHA를 지정하면 데이터 버전을 고정할 수 있습니다. 결과는 실험 UUID별 JSON과 CSV로 저장되며, 문서 수, 질의 수, 모델, 장치, 설정, 각 지표, 검색 latency, indexing/encoding 시간과 메모리 정보를 포함합니다.

아래는 `experiments/e5`와 `experiments/bge-m3`에서 BM25·Dense·Hybrid를 모두 실행한 최신 결과입니다. 두 실행은 2026-11-26에 RTX 3090에서 MTEB AutoRAGRetrieval test split(문서 720개, 질의 114개)을 대상으로 기록했습니다. 질의마다 한 번 warm-up한 뒤 3회 측정해 각 방식당 342개 latency 표본을 얻었습니다. Dense/Hybrid latency에는 질의 임베딩 시간이 포함됩니다. Hybrid 표의 α는 Dense 점수의 가중치입니다.

| 모델 | 방법 | Recall@10 | MRR@10 | nDCG@10 | 평균 검색 시간 | p95 |
|---|---|---:|---:|---:|---:|---:|
| E5 | BM25 | 0.9386 | 0.7944 | 0.8293 | 0.59 ms | 1.20 ms |
| E5 | Dense | 0.9474 | 0.7490 | 0.7975 | 8.90 ms | 10.70 ms |
| E5 | Hybrid, α=0.25 | 0.9825 | 0.8292 | 0.8665 | 9.92 ms | 13.41 ms |
| E5 | Hybrid, α=0.5 | 0.9912 | 0.8694 | 0.8996 | 10.24 ms | 12.68 ms |
| E5 | Hybrid, α=0.75 | 0.9649 | 0.8156 | 0.8526 | 9.82 ms | 11.46 ms |
| BGE-M3 | BM25 | 0.9386 | 0.7944 | 0.8293 | 0.57 ms | 1.24 ms |
| BGE-M3 | Dense | 0.9737 | 0.7835 | 0.8301 | 16.38 ms | 19.77 ms |
| BGE-M3 | Hybrid, α=0.25 | 0.9825 | 0.8344 | 0.8706 | 17.14 ms | 20.88 ms |
| BGE-M3 | Hybrid, α=0.5 | 1.0000 | 0.8885 | 0.9159 | 17.73 ms | 21.19 ms |
| BGE-M3 | Hybrid, α=0.75 | 0.9912 | 0.8611 | 0.8929 | 17.27 ms | 21.38 ms |

이번 실험의 최고 nDCG@10과 Recall@10은 BGE-M3 Hybrid α=0.5에서 기록했습니다. 문서 인코딩 시간은 E5 4.16초, BGE-M3 16.47초였습니다. 결과는 각 실행의 세부 측정값을 그대로 보존한 통합 파일 [최신 비교 JSON](experiments/latest-comparison.json)에서 확인할 수 있습니다. 데이터 `main`은 변경될 수 있으므로 이후 비교에서는 `--revision`에 commit SHA를 지정하는 편이 재현성에 좋습니다.

## FastAPI

BM25 API:

```powershell
uvicorn app.main:app --reload
```

Dense/Hybrid API에서 모델은 앱 시작 시 한 번 로드합니다. 환경 변수로 켭니다.

```powershell
$env:KOSEARCH_ENABLE_DENSE = "true"
$env:KOSEARCH_MODEL = "intfloat/multilingual-e5-base"
$env:KOSEARCH_DEVICE = "cuda"
uvicorn app.main:app --reload
```

Swagger UI는 `/docs`입니다. 지원 API:

- `POST /documents/index`: 문서 전체 인덱스를 교체하고 중복 ID를 검증합니다.
- `POST /search`: `bm25`, `dense`, `hybrid` 중 선택해 검색합니다.
- `POST /evaluate`: 등록된 `autorag` 데이터만 허용하며 결과를 JSON·CSV에 저장합니다.
- `GET /experiments/{experiment_id}`: 저장된 결과를 반환합니다.
- `GET /health`: 서비스, 인덱스, 모델 상태를 반환합니다.

인덱스와 모델별 FAISS 파일은 `data/indexes/`에 저장되고 앱 시작 때 Dense 모델이 켜져 있으면 복원합니다. 실험 JSON/CSV는 `experiments/`에 저장됩니다. 데이터 캐시와 FAISS 인덱스는 `.gitignore` 대상입니다.

## 테스트

```powershell
python -m unittest discover -s tests -v
```

FAISS 저장/복원, hybrid 점수, API 입력 검증과 BM25·metric·dataset schema를 작은 고정 fixture로 검사합니다. 테스트는 모델 가중치를 다운로드하지 않습니다.

