# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:55:07 2021

@author: atr1n17
"""


import json
import spacy 
import codecs
import os
import sys
import difflib
nlp = spacy.load('en_core_web_sm')

def make_data(orig_path, processed_path, split, save_path):      
    with codecs.open(os.path.join(orig_path, split+'.target'), 'r') as f:
        target = f.readlines()
    with codecs.open(os.path.join(orig_path, split+'.source'), 'r') as f:
        source = f.readlines()
    
    data = []
    file_nums = [int(f.split(split+'.')[1].split('.json')[0]) for f in os.listdir(processed_path) if split in f and 'json' in f]
    file_nums.sort()
    for file_num in file_nums:
        file = os.path.join(processed_path, split+'.'+str(file_num)+'.json')
        with open(file, 'r') as f:
            entry = json.load(f)
        data = data+ entry
         
    labels = []
    tokenised_src = []
    for line in source[src_len_start:src_len_stop]:
        #print(len(line))
        src_chars = line.replace(' ','').replace('\n','').lower()
        found = 0
        for entry in data:
            entry_chars = ''.join([''.join([token for token in sent]) for sent in entry['src']])
            entry_chars = entry_chars.replace(' ','').replace('\n','').lower()
            if src_chars == entry_chars:
                labels.append(entry['labels'])
                tokenised_src.append('\n'.join(' '.join(sent) for sent in entry['src']))
                found = 1
                break
        if found == 0:
            print(src_chars[:1000])
            print('----------------')
            entry = data[2]
            entry_chars = ''.join([''.join([token for token in sent]) for sent in entry['src']])
            entry_chars = entry_chars.replace(' ','').replace('\n','').lower()
            output_list = [li for li in difflib.ndiff(src_chars, entry_chars) if li[0] != ' ']   
            print(entry_chars[:1000])
            print(len(src_chars))
            print(len(entry_chars))
            print('****************')
        assert(found == 1) 
                
    
    #with codecs.open(os.path.join(save_path, split+'.json'), 'w') as f:
    #        pass
    i = 0
    for ref_src_line, src_line, tgt_line, label in zip(source, tokenised_src, target, labels): 
        assert (ref_src_line.replace(' ','').replace('\n','').lower() == src_line.replace(' ','').replace('\n','').lower())
        entry = {}
        tgt_doc = nlp(tgt_line)
        tgt_tokens = [[token.text for token in sent] for sent in tgt_doc.sents]
        tgt_text = '\n'.join([' '.join(line) for line in tgt_tokens])
        entry['doc'] = src_line.lower()
        entry['summaries']  = tgt_text.lower()
        entry['labels'] = '\n'.join([str(l) for l in label])
      
        with codecs.open(os.path.join(save_path, split+str(src_len_start)+'.json'), 'a') as f:
            json.dump(entry,f)
            f.write('\n')
            
orig_path = sys.argv[1]
splits = [sys.argv[4]]
save_path = sys.argv[3]
processed_path = sys.argv[2]
src_len_start = int(sys.argv[5])
src_len_stop = int(sys.argv[6])
for split in splits:
    make_data(orig_path, processed_path, split , save_path)
            
