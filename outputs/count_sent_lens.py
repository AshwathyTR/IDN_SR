import sys
path = sys.argv[1]
import os
import math


with open(path, 'r') as f:
    sums = f.read().split('<<END>>')

lens = [len(summ.split('\n')) for summ in sums]
avg = sum(lens) / len(lens)
var = sum((x-avg)**2 for x in lens) / len(lens)
print('mean :'+str(avg))
print('sd :'+str(math.sqrt(var)))
