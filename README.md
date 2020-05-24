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

## Introduction
Space computational costs and time complexities are more and more important in the era of large language models. Therefore ALBERT, a lite BERT, exists. It greatly reduces the time and space needed for training a language model. Furthermore, locality sensitive hashing attention mechanism, which was brought by Reformer, also improves on language model’s space complexity. However, reformer was not implemented on downstream tasks. Therefore, we would like to experiment whether it will achieve a better result if the attention layer for ALBERT is replaced by locality sensitive hashing.
## Model selection
### Locality Sensitive Hashing
Locality Sensitive Hashing is an attention mechanism that replaces the original dot-product attention and reduces the former space complexity of  O(N2) to O(N lg N). It randomly permutes Q vectors several ㄗrounds and hashes each qi into several buckets. This process finds related qis and computes them into attention matrix with lower cost.
### Model building
We connected locality sensitive hashing attention layer on ALBERT structure (fig 1). For training process, we use standard ALBERT pre-train model provided by Google for pre-training, then fine-tune it to specific tasks. The experiment is done on Google Colab with GPU P100.
## Experiment
We experiment our model on natural language understanding datasets (GLUE) for confirmation, then, implement the our model on several tasks such as sentence understanding, sequence generation and text classification. We compare our model with ALBERT to identify the benefit of LSH.  
## Result
## Conclusion
