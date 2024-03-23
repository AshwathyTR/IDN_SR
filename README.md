This repository contains modifications of the SummaRuNNer model from https://github.com/hpzhao/SummaRuNNer to perform Rationale Based Learning for IDN Summarisation.

# Training
```
pipenv run python main.py -alpha_loss 0.50 -epochs 1 -pos_num 3000 -report_every 30 -device 0 -model wordonlyAttnRNNW -save_dir {save_dir} -train_dir {train.json} -val_dir {val.json} -test_dir {test.json} -batch_size 1
```
# Test
```
pipenv run python main.py -test -device 0 -pos_num 3000 -model wordonlyAttnRNNW -batch_size 1 -load_dir {model_path} -test_dir {test.json} -hyp {output_folder} -ref {output_folder}  -topk 81
```

# Evaluation
```
cd outputs
pipenv run eval.py {output_folder}
pipenv run eval_ex.py {output_folder} {oracle_summaries_path}
```

# Contact Info
Feel free to contact atr1n17@soton.ac.uk if you run into any issues or have any questions.
