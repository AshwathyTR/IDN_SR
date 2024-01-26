import sys
path = sys.argv[1]

with open(path, 'r') as f:
   h = f.read().split('<<END>>')

unique_sum = list(set(h))
print("unique summaries: " + str(len(unique_sum)))
print("total summaries: " + str(float(len(h))))
