from rouge_score import rouge_scorer
import sys

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
with open(sys.argv[1],'r') as f:
    hyp = f.read().split('<<END>>')
with open(sys.argv[2],'r') as f:
    ref = f.read().split('<<END>>')

score_sum = 0.0
for h, r in zip(hyp,ref):
    score = scorer.score(h, r)
    score_sum = score_sum +score['rouge1'][2]
print(score_sum/len(hyp))
