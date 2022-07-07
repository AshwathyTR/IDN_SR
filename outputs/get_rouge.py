#!/usr/bin/env python3

import os
from pyrouge import Rouge155
import sys
import shutil
import re
#re.sub('<.*?>', '', string)
def unpack(hyp, ref, out):
    with open(hyp,'r') as f:
        hyp_text = f.read()
        #hyp_text = re.sub('\[\d+]', '', hyp_text)
    with open(ref,'r') as f:
        ref_text = f.read()
        #ref_text = re.sub('\[\d+]', '', ref_text)
    #print (ref_text)
    os.mkdir(os.path.join(out, 'hyp'))
    os.mkdir(os.path.join(out, 'ref'))
    with open(os.path.join(out, 'hyp', '0.txt'),'w') as f:
            f.write(hyp_text.lower())
    with open(os.path.join(out, 'ref', '0.txt'),'w') as f:
            f.write(ref_text.lower())

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
    hyp = sys.argv[1]
    ref = sys.argv[2]
    out = sys.argv[3]
    unpack(hyp, ref, out)
    rouge(out)
    clear(out)
