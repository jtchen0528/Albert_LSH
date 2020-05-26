# Locality Sensitive Hashing on ALBERT
Discuss the impact when locality sensitive hashing from reformer is implemented on ALBERT, a lite BERT.
## Installation
```
pip install transformers
pip install pytorch-reformer
```
replace contents in transformers/modeling_albert.py with contents in ALBERT_LSH.py.
Run each testset.
## run GLUE
```
python download_glue.py
```
```
python run_glue.py \
  --model_type bert \
  --model_name_or_path albert-base-v2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 64 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME/
```
## Introduction
We would like to experiment whether it will achieve a better result if the attention layer for ALBERT is replaced by locality sensitive hashing.
## Model selection
### Locality Sensitive Hashing
Locality Sensitive Hashing is an attention mechanism that replaces the original dot-product attention and reduces the former space complexity of  O(N2) to O(N lg N). It randomly permutes Q vectors several rounds and hashes each qi into several buckets. This process finds related qis and computes them into attention matrix with lower cost.
![snapshot](Files/model.png)
### Model building
We connected locality sensitive hashing attention layer on ALBERT structure (fig 1). For training process, we use standard ALBERT pre-train model provided by Google for pre-training, then fine-tune it to specific tasks. The experiment is done on Google Colab with GPU P100.
![snapshot](Files/train.png)
## Experiment
We experiment our model on natural language understanding datasets (GLUE) for confirmation, then, implement the our model on several tasks such as sentence understanding, sequence generation and text classification. We compare our model with ALBERT to identify the benefit of LSH.  
## Result
MRPC 81.4%
## Conclusion
