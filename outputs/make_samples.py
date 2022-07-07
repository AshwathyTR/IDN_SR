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

    command = '-e /home/atr1n17/pyrouge/tools/ROUGE-1.5.5/data -a -x -d -c 95 -m -n 1'
    output = r.convert_and_evaluate(rouge_args=command)
    output_lines = output.split('\n')[5:] 
    print(output) 
    min_score = 1.0
    max_score = 0.0
    max_id = ''
    min_id = ''
    for line in output_lines:
        if not line.strip():
            continue
        f = float(line.split(' ')[-1].split(':')[1])
        id_ = line.split(' ')[3].split('.')[0]
        print(f)
        print(id_)
        if f>max_score:
            max_score = f
            max_id = id_
        if f<min_score:
            min_score = f
            min_id = id_
    print(max_id)
    print(min_id)
    with open(r.system_dir+'/'+min_id+'.txt','r') as f:
            worst_hyp = f.read()
    with open(r.model_dir+'/'+min_id+'.txt','r') as f:
            worst_ref = f.read()
    with open(r.system_dir+'/'+max_id+'.txt','r') as f:
            best_hyp = f.read()
    with open(r.model_dir+'/'+max_id+'.txt','r') as f:
            best_ref = f.read()
    with open(path+'/worst_hyp.txt','w') as f:
            f.write(worst_hyp)
    with open(path+'/worst_ref.txt','w') as f:
            f.write(worst_ref)
    with open(path+'/best_hyp.txt','w') as f:
            f.write(best_hyp)
    print(max_score)
    print(min_score)
    with open(path+'/best_ref.txt','w') as f:
            f.write(best_ref)

if __name__ == '__main__':
    #remove_broken_files()
    path = sys.argv[1]
    unpack(path)
    rouge(path)
    #clear(path)
