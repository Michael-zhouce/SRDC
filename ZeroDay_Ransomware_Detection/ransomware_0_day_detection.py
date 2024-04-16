from Dataset import Dataset
from LSTM.Model import Classifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import csv
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
    df_train = pd.read_csv(args.Data_Train_path)
    df_train.fillna('', inplace=True)


    df_test = pd.read_csv(args.Data_Test_path)
    df_test.fillna('', inplace=True)
    labels = {
        "Goodware": 0,
        "Critroni": 1,
        "CryptLocker": 2,
        "CryptoWall": 3,
        "KOLLAH": 4,
        "Kovter": 5,
        "Locker": 6,
        "MATSNU": 7,
        "PGPCODER": 8,
        "Reveton": 9,
        "TeslaCrypt": 10,
        "Trojan-Ransom": 11
            }



        
    model = Classifier(hidden_size=768, num_classes=2, max_seq_len=1024, gpt_model_name="RDC-GPT",compression_ratio=128)
    LR = 1e-5
    EPOCHS = 20  
    train(model, df_train, df_test, LR, EPOCHS)
    
def train(model, train_data, test_data, learning_rate, epochs):
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
            input_id = train_input["input_ids"].to(device)
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
        # 验证
        true_labels, pred_labels = evaluate(model, test_data)
        results = pd.DataFrame(classification_report(true_labels,pred_labels,output_dict=True))
        table = PrettyTable()
        table.field_names = [""] + list(results.columns)

        for idx, row in results.iterrows():
            table.add_row([idx] + [round(i, 3) for i in row.tolist () if isinstance(i, float)])
        print(table)
        file_name = "epoch_{}.csv".format(epoch_num + 1)
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
            input_id = test_input['input_ids'].to(device)

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
    parser.add_argument('--Data_Test_path', required=False, help='Test data after internal feature semantic processing csv path', 
                        default=r'test.csv')
    parser.add_argument('--Data_Train_path', required=False, help='Train data after internal feature semantic processing csv path', 
                        default=r'train.csv')
    args=parser.parse_args()
    return args   

if __name__ == "__main__":
    main()