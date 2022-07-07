from rouge_score import rouge_scorer
import sys

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
with open(sys.argv[1]+'/hyp.txt','r') as f:
    hyp = f.read().split('<<END>>')
with open(sys.argv[1]+'/ref.txt','r') as f:
    ref = f.read().split('<<END>>')

score_sum = 0.0
max_score = 0
min_score = 1
skip =0 
for h, r in zip(hyp,ref):
    #if len(r.split('\n'))>150:
    #        skip=skip+1
    #        continue
    score = scorer.score(h, r)['rouge1'][2]
    if score == 0:
        continue
    if score > max_score :
        max_score = score
        max_sum = h
        max_ref = r
        max_id = hyp.index(h)
    if score < min_score:  
        min_score = score
        min_sum = h
        min_ref = r
        min_id = hyp.index(h)

with open(sys.argv[1]+'/max_sum_hyp'+str(max_id)+'_'+str(max_score),'w') as f:
   f.write(max_sum)
with open(sys.argv[1]+'/max_sum_ref'+str(max_score),'w') as f:
   f.write(max_ref)

with open(sys.argv[1]+'/min_sum_hyp'+str(min_id)+'_'+str(min_score),'w') as f:
   f.write(min_sum)
with open(sys.argv[1]+'/min_sum_ref'+str(min_score),'w') as f:
   f.write(min_ref)

#print(skip)

