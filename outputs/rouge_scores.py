from rouge_score import rouge_scorer
import sys

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
with open(sys.argv[1],'r') as f:
    hyp = f.read()
with open(sys.argv[2],'r') as f:
    ref = f.read()
scores = scorer.score(hyp, ref)
print(scores)
