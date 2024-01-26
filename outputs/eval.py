#!/usr/bin/env python3

import os
from pyrouge import Rouge155
import sys
import shutil

def unpack(path):
    with open(path+'/hyp.txt','r') as f:
        hyp = f.read().split('<<END>>')
    with open(path+'/ref.txt','r') as f:
        ref = f.read().split('<<END>>')
    os.mkdir(os.path.join(path, 'hyp'))
    os.mkdir(os.path.join(path, 'ref'))
    for index, entry in enumerate(hyp):
        with open(os.path.join(path, 'hyp', str(index)+'.txt'),'w') as f:
            f.write(entry.lower())
    for index, entry in enumerate(ref):
        with open(os.path.join(path, 'ref', str(index)+'.txt'),'w') as f:
            f.write(entry.lower())

def clear(path):
    shutil.rmtree(os.path.join(path, 'hyp'))
    shutil.rmtree(os.path.join(path, 'ref'))

def rouge(path):
    r = Rouge155()
    r.home_dir = path
    r.system_dir = path+'/hyp'
    r.model_dir =  path+'/ref'

    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = '#ID#.txt'

    command = '-e /home/atr1n17/pyrouge/tools/ROUGE-1.5.5/data -a -x -c 95 -m -n 1'
    output = r.convert_and_evaluate(rouge_args=command)
    print(output)

if __name__ == '__main__':
    #remove_broken_files()
    path = sys.argv[1]
    unpack(path)
    rouge(path)
    clear(path)
