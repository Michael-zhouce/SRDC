# RDCS: Ransomware Detection and Classification Using Semantics with LLM-assisted Pre-training
<h1 align="center">RDCS: Ransomware Detection and Classification Using Semantics with LLM-assisted Pre-training</h1>
<!-- RDCS: Ransomware Detection and Classification Using Semantics with LLM-assisted Pre-training -->


## Feature_Internal_Semantic_Processing

### 1. Download open-source datasets from RISS group <a href="https://github.com/rissgrouphub/ransomwaredataset2016">Link</a>, for fine-tuning and testing models

### 2. Feature Internal Semantic Processing
```bash
python Feature_Internal_Semantic_Processing/Internal_Semantic_Processing.py 
```
Please set the path to the required file
--RansomwareData_csv_path  RansomwareData.csv path
--VariableNames_txt_path  VariableNames.txt path

 An example of usage:
 My RansomwareData_csv_path is data/path_A.csv
 My VariableNames_txt_path is data/path_B.csv

 ```bash
python Feature_Internal_Semantic_Processing/Internal_Semantic_Processing.py --RansomwareData_csv_path data/path_A.csv --VariableNames_txt_path data/path_B.csv
```

## ZeroDay Ransomware Detection

```bash
python ZeroDay_Ransomware_Detection/ransomware_0_day_detection.py  
```
Please set the path to the required file
--Data_Test_path   Training data after internal feature semantic processing
--Data_Train_path  Test data after internal feature semantic processing

So please do the first step of internal feature semantic processing and then divide the training dataset (for fine-tuning) and the test dataset (for testing)

## Ransomware Family Classification

```bash
python Ransomware_Family_Classification/ransomware_family_classifier.py 
```
Please set the path to the required file
--Data_csv_path   Data after internal feature semantic processing

So please do the first step of internal feature semantic processing


## If you have any questions about our project, please send email to 2272127@stu.neu.edu.cn.



