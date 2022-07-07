from bs4 import BeautifulSoup
import sys
path = sys.argv[1]

files = ['best_hyp.txt','best_ref.txt','worst_hyp.txt','worst_ref.txt']
for file in files:
    with open(path+'/'+file,'r') as f:
        htmltxt = f.read()
        soup = BeautifulSoup(htmltxt, features="html.parser")
        txt  = soup.get_text()
    with open(path+'/'+file,'w') as f:
        f.write(txt)    
