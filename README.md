# BERT models vs Siamese LSTM

### Code repository for Masters Thesis titled:

<b>'Transformer Models and their suitability on Semantic Similarity downstream task'</b>

Summary from thesis:

"""

Assessing similarity between sentences is an important task in advancements on how machines process natural language, but also plays a vital role in many businesses. It's applications include i.e. information retrieval, question answering, or duplicated sentence classification. For companies like Quora Inc., or Allegro.pl, such tasks play an essential role for effective business activity. Transformer models stand for most of recent breakthroughs in the field of Natural Language Processing, and achieve state-of-the-art results in most tasks. In this thesis, on one hand there has been an extensive literature overview performed, to provide context of foundations that underpin creation of Transformer models. On the other, there has been an evaluation of transformer based model - BERT - performed on Quora Question Pairs dataset, both from performance, but also from computational efficiency perspective. There were overall two BERT-based models trained, one with siamese architecture, and one with simple classification layer on top of it. As a baseline model there were Siamese Long Short Term Memory Neural Network models trained.

"""

Whole project have been written in PyTorch from scratch, and with use of HuggingFace Transformers. Idea for siameseBERT came from https://arxiv.org/abs/1908.10084.

Link to thesis: ....

All results from BERT models training are available in project:
- https://wandb.ai/ra-v/bert_vs_siameselstm_quora

to install all nessecary packages, assuming you have basic conda configuration installed on your machine, run:
- pip3 install -r requirements.txt
