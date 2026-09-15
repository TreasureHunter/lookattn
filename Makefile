PYTHON_BIN ?= python3
MODEL_PATH ?= models/Qwen2.5-VL-7B-Instruct
SCORER     ?= checkpoints/scorer_best.pt
IMAGE      ?= assets/smoke_test.png
GPU        ?= cuda:0
KEEP_RATIO ?= 0.198
TASK       ?= realworldqa
PRUNING_RATIO ?= 0.8
GPUS       ?= 0

.PHONY: setup setup-vlmeval download smoke-image demo infer train recall \
        prepare-vlmeval evaluate evaluate-vlmeval

## 环境 ---------------------------------------------------------------------

setup:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/setup_runtime.sh

setup-vlmeval:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/setup_vlmevalkit.sh

download:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh scripts/download_model.py \
		--local-dir "$(MODEL_PATH)"

## 推理 ---------------------------------------------------------------------

smoke-image:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh scripts/create_smoke_image.py \
		--output "$(IMAGE)"

# HAWK 原始路径（不需要 checkpoint）
demo: smoke-image
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh scripts/infer.py \
		--model-path "$(MODEL_PATH)" \
		--image "$(IMAGE)" \
		--device "$(GPU)" \
		--keep-ratio "$(KEEP_RATIO)"

# lookahead 路径（用训练好的 scorer）
infer:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh scripts/infer_lookhawk.py \
		--model-path "$(MODEL_PATH)" \
		--scorer "$(SCORER)" \
		--image "$(IMAGE)" \
		--device "$(GPU)" \
		--keep-ratio "$(KEEP_RATIO)"

## 训练 ---------------------------------------------------------------------

train:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh scripts/train_layer1.py \
		--model-path "$(MODEL_PATH)"

## 评测 ---------------------------------------------------------------------

recall:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh scripts/eval_recall_lookhawk.py \
		--model-path "$(MODEL_PATH)" \
		--scorer "$(SCORER)"

prepare-vlmeval:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh \
		scripts/prepare_vlmeval_datasets.py --project-root .

evaluate:
	"$(PYTHON_BIN)" scripts/evaluate.py \
		--task "$(TASK)" \
		--pruning_ratio "$(PRUNING_RATIO)" \
		--model_path "$(MODEL_PATH)" \
		--gpus "$(GPUS)"

evaluate-vlmeval:
	PYTHON_BIN="$(PYTHON_BIN)" CUDA_DEVICES="$(GPUS)" \
		bash scripts/evaluate_vlmeval.sh
