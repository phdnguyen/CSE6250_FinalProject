# CSE 6250: Big Data for Health Informatics 

# Final Project

### Data list

NOTEEVENTS.csv.gz (from MIMIC III Clinical Database 1.4 https://physionet.org/content/mimiciii/1.4/)

GoogleNews-vectors-negative300.bin.gz (https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)

NIHMS1767978-supplement-MIMIC_SBDH.csv (https://pmc.ncbi.nlm.nih.gov/articles/PMC8734043/)

contractions.json (https://www.kaggle.com/datasets/yetman/english-contractions)

### Environment set up

conda create --name `<env>` --file environment.yml

### Directory tree

```
├── README.md
├── data
│   ├── external
│   │   ├── GoogleNews-vectors-negative300.bin.gz
│   │   ├── NIHMS1767978-supplement-MIMIC_SBDH.csv
│   │   └── contractions.json
│   ├── processed
│   │   ├── 0_filtered_data.csv
│   │   ├── 1_cleaned_data.csv
│   │   ├── embedding_matrix.pkl
│   │   ├── input_tensor.pkl
│   │   └── word_index.pkl
│   └── raw
│       └── NOTEEVENTS.csv.gz
├── environment.yml
├── logs
├── models
│   ├── cnn_model_20241110_104156.pt
│   ├── cnn_model_20241110_115610.pt
│   ├── cnn_results_20241110_104156.json
│   ├── cnn_results_20241110_115610.json
│   ├── lstm_best_model.h5
│   └── pytorch_lstm.pt
├── output
└── src
    ├── data_processing
    │   ├── 0_filter_data.py
    │   ├── 1_clean_data.py
    │   └── 2_embedding.py
    ├── models
    │   ├── classic_ml.py
    │   ├── cnn_binary.py
    │   ├── cnn_multiclass.py
    │   ├── lstm_bidirectional.py
    │   ├── lstmu_binary.py
    │   └── lstmu_multiclass.py
    └── utils
        └── plot.py
```

### File Description and How to run

#### All scripts to run the pipeline are in src/data_processing. To process data, run these files sequentially:

0_filter_data.py 
1_clean_data.py
2_embedding.py


#### Each model has its own script in src/models/. Each script will take the processed data and produce results saved in corresponding log files.