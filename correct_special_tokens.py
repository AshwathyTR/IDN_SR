import sys
import os

path = sys.argv[1]
save_path = sys.argv[2]
for file in os.listdir(path):
    with open(os.path.join(path, file),'r') as f:
        text = f.read()
        new_text = text.replace('[EX]', ' [EX] ').replace(':SC:', ' :SC: ').replace('S0:', ' S0: ')
    with open(os.path.join(save_path, file),'w') as f:
        f.write(new_text)

