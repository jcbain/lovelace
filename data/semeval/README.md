# SemEval 2018 Task 1

To use this data cite the following:

```latex
@inproceedings{SemEval2018Task1,
author = {Mohammad, Saif M. and Bravo-Marquez, Felipe and Salameh, Mohammad and Kiritchenko, Svetlana},
title = {SemEval-2018 {T}ask 1: {A}ffect in Tweets},
booktitle = {Proceedings of International Workshop on Semantic Evaluation (SemEval-2018)},
address = {New Orleans, LA, USA},
year = {2018}}
```

Reference this page for data [download](https://competitions.codalab.org/competitions/17751#learn_the_details-datasets) and look under the section **Training, Development, and Test Datasets: For five tasks and three languages** within the bullet *EI-oc*.

## Data
The datasets in this file were downloaded and preprocessed. Raw data and preprocessed files all exist in the `rawdata/` directory. 

file                  | description
--------------------- | -------------------------------------------------------
`train/EI-oc-En*`     | The originally training files for each specific emotion
`train/trainig*.csv`  | The preprocessed files for training
`dev/2018-EI-oc-En*`  | The original development files. Used for training cases
`test/2018-EI-oc-En*` | The testing files

These files each contain data for one of four emotions: anger, fear, sadness and joy. Each of the preprocessed training data sets have been downsampled to the smallest class to ensure class balance when training the model.
