import json
import pickle

def get_json_doc(file_name, index):
    with open(file_name) as f:
        json_list = f.readlines()
    doc = json.loads(json_list[int(index)])['doc']
    return doc

def get_list_from_pkl(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def find_choice_indexes(text):
    lines = text.split("\n")
    choice_indexes = []
    for i, line in enumerate(lines):
        if "choice :" in line:
            choice_indexes.append(i)
    return choice_indexes

def avg_deviation(list1, list2):
    total_deviation = 0
    for num1 in list1:
        closest_num = min(list2, key=lambda x: abs(x-num1))
        deviation = abs(num1 - closest_num)
        total_deviation += deviation
    return total_deviation / len(list1)

import sys
json_file = sys.argv[1]
pkl_file = sys.argv[2]
json_index = sys.argv[3]
json_text = get_json_doc(json_file, json_index)
data_list = get_list_from_pkl(pkl_file)
#print(data_list)
choice_indexes = find_choice_indexes(json_text)
#print(list1)
#print(choice_indexes)
list1 = data_list[int(json_index)][0][:100]
#list1 = data_list[0][0]
#print(data_list[int(json_index)][0])
#print(list1)
#print(choice_indexes)
avg_deviation_val = avg_deviation(list1, choice_indexes)
print(avg_deviation_val)
