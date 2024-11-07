# CSE 6250: Big Data for Health Informatics 

# Final Project

### Data list

NOTEEVENTS.csv.gz (from MIMIC III Clinical Database 1.4 https://physionet.org/content/mimiciii/1.4/)

GoogleNews-vectors-negative300.bin.gz (https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)

NIHMS1767978-supplement-MIMIC_SBDH.csv (https://pmc.ncbi.nlm.nih.gov/articles/PMC8734043/)

contractions.json (https://www.kaggle.com/datasets/yetman/english-contractions)

### Environment set up

conda create --name <env> --file requirements.txt


### Directory tree

.
├── README.md
├── data
│   ├── external
│   │   ├── GoogleNews-vectors-negative300.bin.gz
│   │   ├── NIHMS1767978-supplement-MIMIC_SBDH.csv
│   │   └── contractions.json
│   ├── processed
│   │   ├── 0_filtered_data.csv
│   │   ├── 1_cleaned_data.csv
│   │   ├── embedding_matrix.pkl
│   │   ├── input_tensor.pkl
│   │   └── word_index.pkl
│   └── raw
│       └── NOTEEVENTS.csv.gz
├── logs
│   └── svm_results.log
├── notebook.ipynb
├── output
├── requirements.txt
└── src
    ├── data_preprocessing
    │   ├── 0_filter_data.py
    │   ├── 1_clean_data.py
    │   └── 2_embedding.py
    ├── models
    │   └── cnn.py
    ├── pipelines
    │   ├── lstm.py
    │   └── svm.py
    └── utils
        └── plot.py
