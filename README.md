# Lettuce Dry-Weight Prediction

## Resources

[Info](AIML_Summer_Grand_Challenge_Rules_Updated.pdf)\
[Submission Portal](https://grand-challenge.aiml.team/)\
[Technical Report](technical_report.md)\
[Todo List](TODO.md)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

```text
# provided
data/Test
data/Training

# generated
data/Processed/Test
data/ProcessedTest
```

## Scripts

```text
make preprocess # run preprocessing pipeline
make train # interface your model
make test # generate output csv
```

## Submission

The model predictions are saved in the `outputs/` folder for example 
`outputs/resnet-18/prediction.csv`

Email to: `grand-challenge@aiml.team`

