# Implementation of the Papadamou's model for the paper "Auditing YouTube's Recommendation Algorithm for Misinformation Filter Bubbles"

This repository contains additional supplementary material for the paper *Auditing YouTube's Recommendation Algorithm for Misinformation Filter Bubbles* that has been accepted for publication at ACM TORS journal. The main repository for the paper can be found [here](https://github.com/kinit-sk/yaudit-recsys-2021).

The code presented here is a modification of the [original code](https://github.com/kostantinos-papadamou/pseudoscience-paper) by Kostantinos Papadamou, Savvas Zannettou, Jeremy Blackburn, Emiliano De Cristofaro, Gianluca Stringhini, and Michael Sirivianos for their [paper](https://ojs.aaai.org/index.php/ICWSM/article/view/19329) *"It is just a flu": Assessing the Effect of Watch History on YouTube's Pseudoscientific Video Recommendations*.

### Citing the Paper:
If you make use of any modules available in this repository in your work, please cite the original paper by Papadamou et al.:
```latex
@article{papadamou2020just,
    title={'It is just a flu': Assessing the Effect of Watch History on YouTube's Pseudoscientific Video Recommendations},
    author={Papadamou, Kostantinos and Zannettou, Savvas and Blackburn, Jeremy and De Cristofaro, Emiliano and Stringhini, Gianluca and Sirivianos, Michael},
    journal={arXiv preprint arXiv:2010.11638},
    year={2020}
}
```

If you make use of the modified classification model (any of the three versions provided in this repository, especially if using our dataset) please cite also the following paper:
```latex
@article{srba2022auditing,
    title={Auditing YouTube's Recommendation Algorithm for Misinformation Filter Bubbles},
    author={Srba, Ivan and Moro, Robert and Tomlein, Matus and Pecher, Branislav and Simko, Jakub and Stefancova, Elena and Kompan, Michal and Hrckova, Andrea and Podrouzek, Juraj and Gavornik, Adrain and Bielikova, Maria},
    journal={ACM Transactions on Recommender Systems},
    year={2022}
}
```

## Note on extending the work by Papadamou et al.

In our paper *Auditing YouTube's Recommendation Algorithm for Misinformation Filter Bubbles* we reuse the classification model by Papadamou et al. to automatically annotate videos collected during our audit of YouTube that have not been annotated manually. There are several differencies between theirs and our work which neccessitated modification and extension of the code provided by Papadamou et al. We do not use other parts of the original code, namely the audit framework, although it is included in this repository for the sake of completion.

The main differences and subsequent changes to the classification model are as follows:

* *Different labels (classes)* - the original work used three classes (*science*, *pseudoscience*, and *irrelevant*) which have been collapsed into two used in model training and evaluation (*pseudoscience* and *other*). In our work, we use three different classes (*promoting*, *debunking*, and *neutral*). We experiment with [binary](https://github.com/kinit-sk/yaudit-papadamou-model/tree/main) (either omitting the *neutral* class or merging it with the *debunking* class) as well as with [ternary classification](https://github.com/kinit-sk/yaudit-papadamou-model/tree/three_classes) in which we retain all three classes.
* *Not using tags as input data* - since our collected dataset does not contain this information, we omit tags from model inputs in all our experiments.
* *Adding a variant using BERT embeddings instead of fastText* - besides employing the original version of the model which uses fastText, we also provide a variant using BERT embeddings of video snippets, transcriptions and comments. This is implemented in a separate [`bert-embeddings`](https://github.com/kinit-sk/yaudit-papadamou-model/tree/bert_embeddings) branch.  
* *Different hyperparameters* - for the three classes (ternary) and BERT versions we experimented with different settings of hyperparameters which yielded better results, namely with `LEARNING_RATE`,  `NB_EPOCHS` (number of epochs), and `BATCH_SIZE`. The BERT version also has a different embedding dimension (768 instead of 300) which is due to the embedding size of the used [BERT model](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4).


## How to use this repository

We provide three versions (setups) of the classification model:

1. Binary classification model which is implemented in the [`main`](https://github.com/kinit-sk/yaudit-papadamou-model/tree/main) branch
2. Ternary classification model which is implemented in the [`three-classes`](https://github.com/kinit-sk/yaudit-papadamou-model/tree/three_classes) branch
3. BERT variant of the ternary classification model which is implemented in the [`bert-embeddings`](https://github.com/kinit-sk/yaudit-papadamou-model/tree/bert_embeddings) branch

To use any of the versions, switch to the appropriate branch of the code. Before using any of the versions, please also follow the [installation instructions]((#installation)) by Papadamou et al. If unsure how to proceed, please consult the original [README file](Papadamou_README.md) provided by Papadamou et al. or directly their [repository](https://github.com/kostantinos-papadamou/pseudoscience-paper) for any future updates and bug fixes beyond the scope of our work.


### Table of Contents

- [Installation](#installation)
- [Classification Model](#classification-model)
  - [Prerequisites](#prerequisites)
  - [Training the Classifier](#training-the-classifier)
  - [Classifier Usage](#classifier-usage)
- [LICENSE](#license)


# Installation
Follow the steps below to install and configure all prerequisites for both the training and usage of the classification model. 

### Create and activate Python >=3.6 Virtual Environment
```bash
python3 -m venv --system-site-packages --prompt yaudit-papadamou-model .venv

source .venv/bin/activate
```

### Install required packages
```bash
pip install -r requirements.txt
```

### Install MongoDB
To store the metadata of YouTube videos, as well as for other information we use MongoDB. Install MongoDB on your own system or server using:

- **Ubuntu:** Follow instructions <a href="https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/">here</a>.
- **Mac OS X:** Follow instructions <a href="https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/">here</a>.

### Additional Requirements

#### Install the youtube-dl package
```bash
pip install youtube-dl
```
**Make use of ```youtube-dl``` wisely, carefully sending requests so that you do not spam YouTube with requests and get blocked.**

#### Install Google APIs Client library for Python
This is the library utilized to call the YouTube Data API from Python
```bash
pip install --upgrade google-api-python-client
```

### YouTube Data API
The codebase uses the YouTube Data API to download video metadata. You can enable the YouTube Data API for your Google account and obtain an API key following the steps <a href="https://developers.google.com/youtube/v3/getting-started">here</a>.

Once you have a **YouTube Data API Key**, please set the `YOUTUBE_DATA_API_KEY` variable in the following files:

- `youtubehelpers/config/YouTubeAPIConfig.py`
  

- `youtubeauditframework/utils/YouTubeAuditFrameworkConfig.py` (if you plan to use the auditing framework as well; we do not use it in the context of our work)



# Classification Model
We reuse a model by Papadamou et al. It is a deep learning model geared to detect pseudoscientific YouTube videos, which we employ to distinguish between videos promoting and debunking misinformation. We train and test the modified model using the collected dataset available [here](https://github.com/kinit-sk/yaudit-recsys-2021).

For classifier architecture and description, see the original [README file](Papadamou_README.md).

## Prerequisites
### Download pre-trained fastText word vectors that are fine-tuned during feature engineering on the dataset: 
```bash
cd pseudoscientificvideosdetection/models/feature_extraction

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

unzip wiki-news-300d-1M.vec.zip
```

### Create MongoDB Database and Collections

1. Create a MongoDB database called: `youtube_pseudoscience_dataset` using the terminal.

2. Process the training data which can be acquired in the main [repository](https://github.com/kinit-sk/yaudit-recsys-2021) using the following [notebook](yaudit-data/prepare_data.ipynb). Please note that you will need to get the videos' metadata using the instructions in the main repository. The notebook will output `json` files which can be imported into the MongoDB.

3. Create the following MongoDB collections under the `youtube_pseudoscience_dataset` database that you just created:
- `groundtruth_videos`
- `groundtruth_videos_comments`
- `groundtruth_videos_transcripts`

4. Import prepared data into the collections using the provided [bash script](mongo_scripts.sh).


## Training the Classifier
Classifier training consists of two steps:

1. Fine-tuning fastText models for each video metadata type in case of the binary/ternary variant of the original model. Alternatively, BERT embeddings of the video metadata types are computed (using [`bert_embeddings.ipynb`](https://github.com/kinit-sk/yaudit-papadamou-model/blob/bert_embeddings/bert_embeddings.ipynb)) in case of the BERT version of the model.

2. Training the Pseudoscience deep learning model (we use 10-fold cross-validation). At the end of the training, the best model is stored in `pseudoscientificvideosdetection/models/pseudoscience_model_final.hdf5` or in `pseudoscientificvideosdetection/models/bert/pseudoscience_model_final.hdf5` in case of the BERT version of the model. Additionaly, outputs of the cross-validation are stored in `classifier/training/temp` or in `classifier/training/bert_temp`.

The training is implemented in [`train_script.py`](train_script.py) (no cross-validation) and in [`cv_script.py`](cv_script.py) (with 10-fold cross-validation).


## Classifier Usage
To use the trained classifier, you can use the provided [`classify.py`](classify.py).


# LICENSE

MIT License
