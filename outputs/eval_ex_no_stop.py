#!/usr/bin/env python3

import os
from pyrouge import Rouge155
import sys
import shutil

def unpack(path, oracle_path):
    with open(path+'/hyp.txt','r') as f:
        hyp = f.read().split('<<END>>')
    with open(path+'/ref.txt','r') as f:
        abs_ref = f.read().split('<<END>>')
    with open(oracle_path+'/hyp.txt','r') as f:
        ref = f.read().split('<<END>>')
    with open(oracle_path+'/ref.txt','r') as f:
        abs_ref_o = f.read().split('<<END>>')

    os.mkdir(os.path.join(path, 'hyp'))
    os.mkdir(os.path.join(path, 'ref'))
    for index, entry in enumerate(hyp):
        if not entry.strip():
            continue
        with open(os.path.join(path, 'hyp', str(index)+'.txt'),'w') as f:
            f.write(entry.lower())
        ex_ref = ref[index]
        o_abs =  abs_ref_o[index].lower().replace(' ','').replace('\n','')
        c_abs = abs_ref[index].lower().replace(' ','').replace('\n','')
        if not (o_abs == c_abs):
             print (index)
             print (o_abs)
             print('_____________________________________')
             print(c_abs)
        assert (o_abs == c_abs)
        with open(os.path.join(path, 'ref', str(index)+'.txt'),'w') as f:
             f.write(ex_ref.lower())


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

    command = '-e /home/atr1n17/pyrouge/tools/ROUGE-1.5.5/data -a -s -x -c 95 -m -n 1'
    output = r.convert_and_evaluate(rouge_args=command)
    output_dict = r.output_to_dict(output)
    return output_dict["rouge_1_f_score"]


if __name__ == '__main__':
    #remove_broken_files()
    #datasets = ['cnn_dm','novel','crd3_corrected_tokens','sb_corrected_tokens','idn_corrected_tokens']
    #models = ['rand','lead','tr_','bs_','sr_','lf_3']
    #lens = [3, 9, 27, 81]
    datasets = ['sb_corrected_tokens']
    models = ['tr_']
    lens = [3, 9, 27, 81]
    paths = []
    oracle_paths = []
    for dataset in datasets:
        for model in models:
             for l in lens:
                 oracle_path = dataset+'/'+'oracle'+str(l) 
                 path =dataset+'/'+model+str(l)
                 paths.append(path)
                 oracle_paths.append(oracle_path)
    scores = {}
    for path, oracle_path in zip(paths, oracle_paths):
         if not os.path.exists(path):
             continue
         try:
             unpack(path, oracle_path)
         
             score = rouge(path)
             scores[path] = score
         except:
         
             #clear(path)
             continue
         #clear(path)
    print(scores)
