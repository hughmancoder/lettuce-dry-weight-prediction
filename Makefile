PYTHON = python3

install:
	pip install -r requirements.txt

preprocess:
	$(PYTHON) src/preprocessing.py

train:
	$(PYTHON) src/train.py

test:
	$(PYTHON) src/inference.py

clean:
	rm -rf data/Processed
	rm -f *.pth
	rm -f src/submission.csv
	find . -name "__pycache__" -exec rm -rf {} +