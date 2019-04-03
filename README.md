# Ada **Lovelace** 

Neural networks for sentiment analysis and phrase classification.

**NOTE:** To replicate the training and classification process of the article *Analyzing the Geographic Relationship of Angry Immigration Tweets Classified by a Gated Recurrent Unit*, following the replication steps in the `docs/training_anger_model.ipynb` notebook.

## Installation
#### Docker
You can build the docker image
```bash
cd /path/to/lovelace
docker build -t lovelace .
```

..and then starting a container in interactive mode

```bash
docker container run -it lovelace 
```

#### Downloading dependencies
If you don't download the docker image (recommended) then you can ensure you have the corrected dependencies by running the `requirements.txt`. If going this route, I would recommend a virtualenv or conda env before downloading. 

```bash
cd /path/to/lovelace
pip3 -r requirements.txt
```


### Training an Anger Model 
To train an anger model, using the modified [SemEval 2018 Task 1 data](https://competitions.codalab.org/competitions/17751), and using pre-trained [GLoVe word vectors](https://nlp.stanford.edu/projects/glove/), run the following in a terminal. 
```bash
python train_scully.py --data_file data/semeval/rawdata/train/training_anger_tri_class.csv  --embedding_file ~/Dropbox/tweets/model_data/pretrained_embeddings/twitter100.npz  --test_size 0.15 --batch_size 128 --hidden_dim 120 --num_classes 3 --num_epochs 5000 --learning_rate 0.00001 --embed_dim 100 --cell_type "lstm" --num_layers 2
```

