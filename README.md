# Lettuce Dry-Weight Prediction

## Resources

[Info](AIML_Summer_Grand_Challenge_Rules_Updated.pdf)\
[Dataset Description](https://grand-challenge.aiml.team/details.html)\
[Submission Portal](https://grand-challenge.aiml.team/)\
[Technical Report](technical_report.md)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Add the dataset to the `data/` folder. It can be downloaded [here](https://uao365.sharepoint.com/sites/AIMLEngineering/Shared%20Documents/Forms/AllItems.aspx?CT=1762817515016&OR=OWA%2DNT%2DMail&CID=3605defb%2Dbab0%2De3c9%2D6b49%2Df844eef63574&e=5%3A26ea1c8041b744cb81d83bb050aff6ce&sharingv2=true&fromShare=true&at=9&FolderCTID=0x0120009ACB622FFCE71644BD3C1327DA2C39AF&id=%2Fsites%2FAIMLEngineering%2FShared%20Documents%2FProjects%2FAIML%20Grand%20Discovery%20Challenge%2F2025%2FGrand%20Challenge%20%2D%20Lettuce%2FFor%20Participants)

The dataset is structured as follows

```text
# provided
data/Test
data/Training

# generated via make preprocess
data/Processed/Test
data/Processed/Train
```

## Scripts

```text
make preprocess # run preprocessing pipeline
make train
make test
make pipeline 
```

## Submission

The model predictions are saved in the `outputs/` folder for example 
`outputs/resnet-18/prediction.csv`

Email to: `grand-challenge@aiml.team`
