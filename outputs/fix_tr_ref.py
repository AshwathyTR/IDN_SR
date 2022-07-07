tr_folders = ['tr_3','tr_9','tr_27','tr_81']

import sys

path = sys.argv[1]

import os
from nltk.tokenize import sent_tokenize

for folder in tr_folders:
    ref_path = os.path.join(path, folder, 'ref.txt')
    with open(ref_path, 'r') as f:
        refs = f.read().split('<<END>>')
    for ref in refs:
        ref = '\n'.join(sent_tokenize(ref))
        with open(ref_path+'.1','a') as f:
            f.write(ref+'<<END>>')
            
    
