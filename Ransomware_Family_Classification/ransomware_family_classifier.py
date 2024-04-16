import sys
import os
from Dataset import Dataset
from LSTM.Model import Classifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import argparse
from torch import nn 
from torch.optim import Adam
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm 
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import random
from prettytable import PrettyTable

def main() :
    args = get_args()
    df = pd.read_csv(args.Data_csv_path)
    df.fillna('', inplace=True)
    EPOCHS = 10  
    LR = 1e-5
    kfold_cross_validation(df,LR, EPOCHS, k_folds=4)

def kfold_cross_validation(df, LR, EPOCHS, k_folds):
   
    skf = StratifiedKFold(n_splits=k_folds,shuffle=True,random_state=42)
    X = df.drop('family', axis=1)
    y = df['family']
    fold = 0
    for train_idx, test_idx in skf.split(X, y):
        fold += 1
        model = Classifier(hidden_size=768, num_classes=12, max_seq_len=1024, gpt_model_name="/home/z50036508/pooling_model/gpt2", compression_ratio=1024)
        print(f"Fold {fold}/{k_folds}")
        
        with open('result.txt', 'a') as f:
            print(f"Fold {fold}/{k_folds}",file=f)
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        train(model, df_train, df_test, LR, EPOCHS,fold)

    

def train(model, train_data, test_data, learning_rate, epochs, fold):
    train = Dataset(train_data)
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)

    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            model.zero_grad()
            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1)==train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
    
        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} ")
        with open('result.txt', 'a') as f:
            print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f}", file=f)
        # Evaluation
        true_labels, pred_labels = evaluate(model, test_data)
        results = pd.DataFrame(classification_report(true_labels,pred_labels,output_dict=True))
        table = PrettyTable()
        table.field_names = [""] + list(results.columns)

        for idx, row in results.iterrows():
            table.add_row([idx] + [round(i, 3) for i in row.tolist () if isinstance(i, float)])
        print(table)
        file_name = "fold_{}_epoch_{}.csv".format(fold, epoch_num + 1)
        table2csv(table.get_string(),file_name)
        with open('result.txt', 'a') as f:
            print(table, file=f)
        
def table2csv(string_data, table_name):
    # Parses the text and converts it to a list
    lines = [line.strip() for line in string_data.strip().split('\n')]
    # Delete the + signs at the beginning and end
    lines = [line[1:-1] for line in lines]

    # Write data to CSV file
    with open(table_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for line in lines:
            row = [cell.strip() for cell in line.split('|')]
            # Delete empty string and write to CSV file
            writer.writerow([cell for cell in row if cell])

# Evaluation
def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

         
    # Tracking variables
    predictions_labels = []
    true_labels = []
    
    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
            # add original labels
            true_labels += test_label.cpu().numpy().flatten().tolist()
            # get predicitons to list
            predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()
    with open('result.txt', 'a') as f:
           print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}', file=f)
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return true_labels, predictions_labels
def get_args():
    parser =argparse.ArgumentParser(description='Description of the parameters of the program run command')

    ## fine-tuning data set 
    parser.add_argument('--Data_csv_path', required=False, help='RansomwareData.csv path', 
                        default=r'formatted_data_split_feature.csv')
    args=parser.parse_args()
    return args
   

if __name__ == "__main__":
    main()