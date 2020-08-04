# SentiHoodModels
This repository contains the implementation of BERT models for Targeted Aspect-based Sentiment Analysis (TABSA), trained and tested on the SentiHood Dataset. The models implemented were proposed in the paper [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/pdf/1903.09588.pdf).

Specifically, this repository contains five BERT models for TABSA, i.e. BERT-single, BERT-pair QA-M, BERT-pair NLI-M, BERT-pair QA-B, and BERT-pair NLI-B. Data and models could be accessed and downloaded from this [google drive](https://drive.google.com/drive/folders/13s4bJLXv6YqlzETg0qSz4jqqf6jxWmm_?usp=sharing).

## Project Structure
<pre>
.
├── Bert-pair
│   ├── NLI-B
│   │   ├── BERT_pair_NLI_B.ipynb
│   │   ├── Datasets
│   │   │   ├── testing_set.csv
│   │   │   ├── training_set.csv
│   │   │   └── validation_set.csv
│   │   ├── Inference_and_Evaluation_of_BERT_pair_NLI_B.ipynb
│   │   ├── Models
│   │   │   ├── accuracy.txt
│   │   │   └── loss.txt
│   │   ├── PredictedValues.csv
│   │   └── preds.jsonl
│   ├── NLI-M
│   │   ├── BERT-pair NLI-M.ipynb
│   │   ├── Datasets
│   │   │   ├── testing_set.csv
│   │   │   ├── training_set.csv
│   │   │   └── validation_set.csv
│   │   ├── Inference and Evaluation of BERT-pair NLI-M.ipynb
│   │   ├── Models
│   │   │   ├── accuracy.txt
│   │   │   └── loss.txt
│   │   ├── PredictedValues.csv
│   │   └── pred.jsonl
│   ├── QA-B
│   │   ├── BERT-pair QA-B.ipynb
│   │   ├── Datasets
│   │   │   ├── testing_set.csv
│   │   │   ├── training_set.csv
│   │   │   └── validation_set.csv
│   │   ├── Inference_and_Evaluation_of_BERT_pair_QA_B.ipynb
│   │   ├── Models
│   │   │   ├── accuracy.txt
│   │   │   └── loss.txt
│   │   ├── PredictedValues.csv
│   │   └── preds.jsonl
│   └── QA-M
│       ├── BERT-pair QA-M.ipynb
│       ├── Datasets
│       │   ├── testing_set.csv
│       │   ├── training_set.csv
│       │   └── validation_set.csv
│       ├── Inference and Evaluation of BERT-pair QA-M.ipynb
│       ├── Models
│       │   ├── accuracy.txt
│       │   └── loss.txt
│       ├── PredictedValues.csv
│       └── preds.jsonl
├── Bert-single
│   ├── Bert-single Model.ipynb
│   ├── Inference and Evaluation of BERT-single.ipynb
│   ├── LocationAspectModels
│   │   ├── LOCATION1dining
│   │   │   ├── accuracy.txt
│   │   │   └── loss.txt
│   │   ├── ...
│   ├── PredictedData
│   │   ├── PredictedLOCATION1dining.csv
│   │   ├── ...
│   ├── preds.jsonl
│   ├── TestingData
│   │   ├── LOCATION1dining.csv
│   │   ├── ...
│   ├── TrainingData
│   │   ├── LOCATION1dining.csv
│   │   ├── ...
│   └── ValidationData
│       ├── LOCATION1dining.csv
│       ├── ...
├── Data Pre-processing New.ipynb
├── README.md
├── Result Analysis.pdf
└── SentiHood Dataset
    ├── sentihood-dev.json
    ├── sentihood-test.json
    └── sentihood-train.json

44 directories, 189 files
</pre>

**Data Pre-processing New.ipynb**: This notebook contains the code for generating datasets (training, validation, andtesting).

**SentiHood Dataset**: This directory contains the SentiHood dataset downloaded from mirror [link](https://github.com/uclmr/jack/tree/master/data/sentihood).

Each BERT-pair model directory has the following structure:
- **Datasets**: This directory contains the training, validation and testing set.
- **Models**: This directory is suppose to contain trained models, as well as their training loss and accuracy logs. Due to GitHub's file size limit, the models are not available in this repository, but can be downloaded using the google drive link provided above.
- **BERT_pair_\*.ipynb**: This notebook contains the implementation for training and validating the models.
- **Inference_and_Evaluation_of_BERT_pair_\*.ipynb**: This notebook contains the code for inference on the testing set, evaluation and construction of `preds.jsonl`.
- **PredictedValues.csv**: This csv file contains the predicted and true labels of testing set which are used for evaluation purposes.
- **preds.jsonl**: This is a json lines file containing both the annotated aspects and sentiments as well as model predictions. Each line is a valid JSON containing data related to single opinion.

BERT-single directory structure:
- **TrainingData**: This directory contains the training data of location-aspect pair models.
- **ValidationData**: This directory contains the validation data of location-aspect pair models.
- **TestingData**: This directory contains the testing data of location-aspect pair models.
- **LocationAspectModels**: This directory is suppose to contain trained location-aspect pair models, as well as their training loss and accuracy logs. Due to GitHub's file size limit, the models are not available in this repository, but can be downloaded using the google drive link provided above.
- **PredictedData**: This directory contains the predictions of location-aspect pair models on the testing set. This is mainly used for evaluation purposes.
- **Bert-single Model.ipynb**: This notebook contains the implementation for training and validating location-aspect pair models.
- **Inference and Evaluation of BERT-single.ipynb**: This notebook conatins the code for inference on the testing set, evaluation and construction of `preds.jsonl`.
- **preds.jsonl**: TThis is a json lines file containing both the annotated aspects and sentiments as well as model predictions. Each line is a valid JSON containing data related to single opinion.

## Requirements
All the notebooks in this repository have been implemented and executed on Google Colab. Hence they could be directly utilized with the same. However, to execute them on local machine, you will need to have following requirements fulfilled:
- Python
- Pytorch
- Transformers
- Scikit-learn
- Pandas
- Numpy
- Matplotlib

## Steps for execution
1. **Generate Datasets**: Utilize the Data Pre-processing New.ipynb notebook to generate the required training, validation and testing set for the models.
2. **Train Models**: Utilize the appropriate notebook to train and validate models.
3. **Select Best Model**: Every notebook, which is used to train models, computes and stores the accuracy of the model on validation test after every epoch. Use this accuracy to manual select the best model and rename it to `best.bin` in the same directory.
4. **Inference and Evaluation**: Utilize the inference and evaluation notebook for making inferences as well as evaluation on the testing set using the best trained model.

## Evaluation Results
The following evaluation results are based on all aspects of the SentiHood dataset, as opposed to just the top four most frequent aspects described in the original paper.

|                 | Sentiment Accuracy | Aspect Accuracy | Aspect F1 score |
|:---------------:|:------------------:|:---------------:|:---------------:|
|   BERT-single   |        96.45       |       68.6      |      88.19      |
|  **BERT-pair QA-M** |        **97.3**        |      **75.04**      |      **91.11**      |
| BERT-pair NLI-M |        97.28       |      74.77      |      90.86      |
|  BERT-pair QA-B |        96.87       |      72.38      |       89.5      |
| BERT-pair NLI-B |        96.66       |      71.47      |      89.51      |

*Note*: While executing the notebooks, you may need to change the existing path based on your system. And contributions are most welcomed. You can either open an issue or directly contact [me](nix07.github.io). 
