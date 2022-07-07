import json
import sys

split = sys.argv[1]
path = split+'.json'

with open(path,'r') as f:
    h = f.readlines()



with open(split+'_r.json','w') as f:
    pass


for line in h:
    j = json.loads(line)
    s = j['doc'].split('\n')
    r = []
    for line in s:
        if 'choice:' in line.lower():
            r.append('1')
        else:
            r.append('0')
    r = '\n'.join(r)
    j['rationale'] = r
    with open(split+'_r.json','a') as f:
        json.dump(j,f)
        f.write('\n')

