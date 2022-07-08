import json
import sys
import os

dpath = sys.argv[1]
split = sys.argv[2]
look_behind = int(sys.argv[3])
look_ahead  = int(sys.argv[4])
path = os.path.join(dpath, split+'.json')

with open(path,'r') as f:
    h = f.readlines()



with open(path[:-5]+'_r.json','w') as f:
    pass


for text in h:
    j = json.loads(text)
    s = j['doc'].split('\n')
    r = []
    for index in range(0,len(s)):
        window = '\n'.join(s[max(0,index - look_behind) : index + look_ahead])
        #print(index)
        #print(window)
        #print('..............')
        
        if 'choice :' in window.lower():
            r.append('1')
            #print(index)
        else:
            r.append('0')
    r = '\n'.join(r)
    j['rationale'] = r
    #print(r)
    with open(path[:-5]+'_r.json','a') as f:
        json.dump(j,f)
        f.write('\n')

