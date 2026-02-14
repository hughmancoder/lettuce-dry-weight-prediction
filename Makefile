PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

TRAIN_CSV := data/Training/Train.csv
TEST_CSV := data/Test/final.csv
PROCESSED_ROOT := data/Processed
TRAIN_PROCESSED := $(PROCESSED_ROOT)/Train
TEST_PROCESSED := $(PROCESSED_ROOT)/Test

# Hybrid Model Paths
HYBRID_WEIGHTS_DIR := weights/hybrid
HYBRID_OUTPUT_DIR := outputs/hybrid
HYBRID_PREDICTION_FILE := $(HYBRID_OUTPUT_DIR)/prediction.csv

# LettuceNet MTL Model Paths (Renamed from Simple)
MTL_WEIGHTS_DIR := weights/lettuce_net_mtl
MTL_OUTPUT_DIR := outputs/lettuce_net_mtl
MTL_PREDICTION_FILE := $(MTL_OUTPUT_DIR)/prediction.csv

.PHONY: install preprocess \
    train test pipeline \
    train-hybrid test-hybrid pipeline-hybrid \
    train-mtl test-mtl est-mtl pipeline-mtl \
    clean

install:
	$(PIP) install -r requirements.txt

preprocess:
	$(PYTHON) src/preprocessing.py \
		--train-csv $(TRAIN_CSV) \
		--test-csv $(TEST_CSV) \
		--processed-root $(PROCESSED_ROOT)



train:
	$(PYTHON) src/train_lettuce_net_mtl.py \
		--train-csv $(TRAIN_CSV) \
		--processed-root $(TRAIN_PROCESSED) \
		--weights-dir $(MTL_WEIGHTS_DIR) \
		--output-dir $(MTL_OUTPUT_DIR)

test:
	$(PYTHON) src/inference_mtl.py \
		--test-csv $(TEST_CSV) \
		--processed-root $(TEST_PROCESSED) \
		--weights-dir $(MTL_WEIGHTS_DIR) \
		--output-file $(MTL_PREDICTION_FILE)

pipeline: preprocess train test



clean:
	rm -rf data/Processed
	rm -rf $(HYBRID_WEIGHTS_DIR)
	rm -rf $(MTL_WEIGHTS_DIR)
	rm -f $(HYBRID_PREDICTION_FILE)
	rm -f $(MTL_PREDICTION_FILE)
	find . -name "__pycache__" -exec rm -rf {} +
