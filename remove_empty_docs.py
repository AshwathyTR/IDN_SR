import json
import os
import sys
path = sys.argv[1]
splits = ['train','test','val']
for split in splits:
    with open(os.path.join(path, split+'_corrected.json'),'r') as f:
        h = f.readlines()
    with open(os.path.join(path, split+'_corrected_1.json'),'w') as f:
        pass

    for line in h:
        data = json.loads(line)
        if data['doc']:
            with open(os.path.join(path, split+'_corrected_1.json'),'a') as f:
                json.dump(data, f)
                f.write('\n')
