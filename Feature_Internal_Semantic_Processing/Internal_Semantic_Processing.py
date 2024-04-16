import os
import pandas as pd
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


def reformat_registry_text(input_text):
    if input_text.startswith("REG:OPENED:"):
        content = input_text[len("REG:OPENED:"):]
        return "opened registry " + content
    elif input_text.startswith("REG:READ:"):
        content = input_text[len("REG:READ:"):]
        return "read registry " + content
    elif input_text.startswith("REG:WRITTEN:"):
        content = input_text[len("REG:WRITTEN:"):]
        return "wrote registry " + content
    elif input_text.startswith("REG:DELETED:"):
        content = input_text[len("REG:DELETED:"):]
        return "deleted registry " + content
    else:
        return "Unknown registry action"


def reformat_drop_text(input_text):
    if input_text.startswith("DROP:"):
        extension = input_text[len("DROP:"):]
        return f"dropped file's extension involved {extension}"
    else:
        return "Invalid drop action"


def reformat_files_ext_text(input_text):
    if input_text.startswith("FILES_EXT:WRITTEN:"):
        content = input_text[len("FILES_EXT:WRITTEN:"):]
        formatted_action = "writing to"
    elif input_text.startswith("FILES_EXT:READ:"):
        content = input_text[len("FILES_EXT:READ:"):]
        formatted_action = "reading"
    elif input_text.startswith("FILES_EXT:OPENED:"):
        content = input_text[len("FILES_EXT:OPENED:"):]
        formatted_action = "opening"
    else:
        return "Unknown file action"
    return f"operations involved {formatted_action} file with extension {content}"

def parseToSentence(input):
    out = ''
    pre = -1
    for i in range(0, len(input)):
        if input[i].isupper():
            out = out + input[pre + 1 : i] + ' ' + input[i].lower()
            pre = i
    if pre != len(input) - 1:
        out = out + input[pre + 1 : len(input)]
    return out.strip()

def reformat_api_text(input_text):
    input_text = input_text[4:]
    if (
        input_text.endswith("A")
        or input_text.endswith("W")
        or input_text.endswith("\n")
    ):
        input_text = input_text[0: len(input_text) - 1]
    if input_text.endswith("Ex"):
        input_text = input_text[0: len(input_text) - 2]
    sentence = parseToSentence(input_text)
    if sentence.startswith("nt"):
        sentence = sentence.replace("nt", "kernel", 1)
    if "__" in sentence:
        sentence = sentence.replace("__", " ")
    return "API:"+sentence


def parseDIR(origin_str):
    delimiter = ':'
    prefix=""
    result = origin_str.split(delimiter)
    if result[1]=='ENUMERATED':
         prefix= "enumerated directory "
    elif result[1]=='CREATED':
         prefix= "created directory "
    path_way=""
    result=result[2:]
    for str in result:
        path_way=path_way+str
    return prefix+path_way


def parseSTR(origin_str):
    result = origin_str.split(':')[1].split(';')[1]
    return "embeded string "+result


def parseFILES(origin_str):
    delimiter = ':'
    prefix=""
    result = origin_str.split(delimiter)
    if result[1]=='OPENED':
         prefix= "opened file in "
    elif result[1]=='READ':
         prefix= "read file in "
    elif result[1]=='WRITTEN':
         prefix= "wrote file in "
    path_way=""
    result=result[2:]
    for str in result:
        path_way=path_way+str    
    return (prefix+path_way)





def main():
    args = get_args()
    raw_data_map = {} 
    with open(args.VariableNames_txt_path, 'r') as file:
        for line in file :
                parts = line.strip().split(';', 1)
                key = int(parts[0])  # KEY  12
                value = parts[1]    #"API:NtOpenFile"
                raw_data_map[key] = value
    # reformat the future
    for key in sorted(raw_data_map.keys()):
            if key in range(4, 236):
                raw_data_map[key] = reformat_api_text(raw_data_map[key])
            if key in range(236, 582):
                raw_data_map[key] = reformat_drop_text(raw_data_map[key])
            if key in range(582, 7204):
                raw_data_map[key] = reformat_registry_text(raw_data_map[key])
            if key in range(7204, 11345):
                raw_data_map[key] = parseFILES(raw_data_map[key])
            if key in range(11345, 12280):
                raw_data_map[key] = reformat_files_ext_text(raw_data_map[key])
            if key in range(12280, 14704):
                raw_data_map[key] = parseDIR(raw_data_map[key])
            if key in range(14704, 30971):
                raw_data_map[key] = parseSTR(raw_data_map[key])


    # raw data import
    df = pd.read_csv(args.RansomwareData_csv_path, header = None)

    # 每一列
    apiFeatures = []
    dropFeatures = []
    regFeatures = []
    filesFeatures =[]
    filesEXTFeatures = []
    dirFeatures = []
    strFeatures = []

    # 相同特征拼接为一个字符串
    # Iterate through rows of the DataFrame
    for index, row in df.iterrows():
        # 每一个样本
        string_parts = ['','','','','','','']  # Store translation strings for this row
        feature_number = 0
        # Iterate through columns after the 3rd column (index 2)
        for col_num, value in enumerate(row.values[3:], start=3):
            if value == 1 and col_num in raw_data_map and col_num in range(235):
                string_parts[0]=string_parts[0]+raw_data_map[col_num+1]+'. '
            elif value == 1 and col_num in raw_data_map and col_num in range(235, 581):
                string_parts[1]=string_parts[1]+raw_data_map[col_num+1]+'. '
            elif value == 1 and col_num in raw_data_map and col_num in range(581, 7203):
                string_parts[2]=string_parts[2]+raw_data_map[col_num+1]+'. '
            elif value == 1 and col_num in raw_data_map and col_num in range(7203, 11344):
                string_parts[3]=string_parts[3]+raw_data_map[col_num+1]+'. '
            elif value == 1 and col_num in raw_data_map and col_num in range(11344, 12279):
                string_parts[4]=string_parts[4]+raw_data_map[col_num+1]+'. '
            elif value == 1 and col_num in raw_data_map and col_num in range(12279, 14703):
                string_parts[5]=string_parts[5]+raw_data_map[col_num+1]+'.'
            elif value == 1 and col_num in raw_data_map and col_num in range(14703,30970):
                string_parts[6]=string_parts[6]+raw_data_map[col_num+1]+'. '
        apiFeatures.append(string_parts[0])
        dropFeatures.append(string_parts[1])
        regFeatures.append(string_parts[2])
        filesFeatures.append(string_parts[3])
        filesEXTFeatures.append(string_parts[4])
        dirFeatures.append(string_parts[5])
        strFeatures.append(string_parts[6])
    # Add the generated strings to the DataFrame as a new column


    formatted_df = df[[2]]
    formatted_df['apiFeatures'] = apiFeatures
    formatted_df["dropFeatures"] = dropFeatures
    formatted_df["regFeatures"] = regFeatures
    formatted_df["filesFeatures"] = filesFeatures
    formatted_df["filesEXTFeatures"] = filesEXTFeatures
    formatted_df["dirFeatures"] = dirFeatures
    formatted_df["strFeatures"] = strFeatures
    formatted_df.rename(columns={2: 'family'}, inplace=True)
    formatted_df.to_csv("after_feature_internal_semantic_process_data.csv", index=False)

def get_args() :
    parser =argparse.ArgumentParser()
    parser.add_argument('--RansomwareData_csv_path', required=False, help='RansomwareData.csv path', 
                        default=r'/home/z50036508/ransomwaredataset2016-main/RansomwareData.csv')
    parser.add_argument('--VariableNames_txt_path', required=False, help='VariableNames.txt path', 
                        default=r'/home/z50036508/ransomwaredataset2016-main/VariableNames.txt')
    args=parser.parse_args()
    return args


if __name__ == "__main__":
    main()