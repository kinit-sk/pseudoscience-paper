#!/usr/bin/python

from dataset.DatasetUtils import DatasetUtils
import fasttext
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from joblib import Parallel, delayed


class BertDataPreparation(object):
    """
    Class that prepares all type of input features required for the training of our
    Pseudoscience Videos Detection Classifier
    """
    def __init__(self, dataset_object):
        # Set Base Directories
        self.INPUT_FEATURES_DIR = 'dataset/data/bert_inputs'

        # Get Dataset Object
        self.DATASET = dataset_object
        self.video_ids = self.DATASET.get_groundtruth_videos()
        return

    def get_ordered_input_features(self, input_features):
        input_features_ordered = {}

        for video_id in self.video_ids:
            if video_id not in input_features:
                raise KeyError('Video id missing')
            
            input_features_ordered[video_id] = input_features[video_id].numpy()[0]
        
        return input_features_ordered

    def get_video_snippet_model_input_features(self, overwrite=False):
        """
        Method that generates an Embedding Vector of each video Snippet (video title + video description)
        :param overwrite: True if existing pre-generated features should be re-generated, False if not
        :return:
        """
        # Create Video Snippet Features filename
        video_snippet_features_filename = '{0}/video_snippets.pickle'.format(self.INPUT_FEATURES_DIR)

        print('\n--- Getting VIDEO SNIPPET features embeddings using BERT uncased model')
        if not os.path.isfile(video_snippet_features_filename) or overwrite:
            raise FileNotFoundError('Video snippet BERT embeddings missing')
        else:
            # Return pre-generated input features
            video_snippet_features = pickle.load(open(video_snippet_features_filename, mode='rb'))
            video_snippet_features = self.get_ordered_input_features(video_snippet_features)

        return list(video_snippet_features.values())

    def get_video_tags_model_input_features(self, overwrite=False):
        """
        Method that generates an Embedding Vector of each Video's Tags
        :param overwrite: True if existing pre-generated features should be re-generated, False if not
        :return:
        """
        # Create Video Tags Features filename
        video_tags_features_filename = '{0}/video_tags.pickle'.format(self.INPUT_FEATURES_DIR)

        print('\n--- Getting VIDEO TAGS features embeddings using BERT uncased model')
        if not os.path.isfile(video_tags_features_filename) or overwrite:
            raise FileNotFoundError('Video tags BERT embeddings missing')
        else:
            # Return pre-generated input features
            video_tags_features = pickle.load(open(video_tags_features_filename, mode='rb'))
            video_tags_features = self.get_ordered_input_features(video_tags_features)
        return list(video_tags_features.values())

    def get_video_transcript_model_input_features(self, overwrite=False):
        """
        Method that generates an Embedding Vector of each video's Transcript
        :param overwrite: True if existing pre-generated features should be re-generated, False if not
        :return:
        """
        # Create Video Transcripts Features filename
        video_transcript_features_filename = '{0}/video_transcripts.pickle'.format(self.INPUT_FEATURES_DIR)

        print('\n--- Getting VIDEO TRANSCRIPT features embeddings using BERT uncased model')
        if not os.path.isfile(video_transcript_features_filename) or overwrite:
            raise FileNotFoundError('Video transcripts BERT embeddings missing')
        else:
            # Return pre-generated input features
            video_transcript_features = pickle.load(open(video_transcript_features_filename, mode='rb'))
            video_transcript_features = self.get_ordered_input_features(video_transcript_features)
        return list(video_transcript_features.values())

    def get_video_comments_model_input_features(self, overwrite=False):
        """
        Method that generates an Embedding Vector of each Video's Comments
        :param overwrite: True if existing pre-generated features should be re-generated, False if not
        :return:
        """
        # Create Video Comments Features filename
        video_comments_features_filename = '{0}/video_comments.pickle'.format(self.INPUT_FEATURES_DIR)

        print('\n--- Getting VIDEO COMMENTS features embeddings using BERT uncased model')
        if not os.path.isfile(video_comments_features_filename) or overwrite:
            raise FileNotFoundError('Video comments BERT embeddings missing')
        else:
            # Return pre-generated input features
            video_comments_features = pickle.load(open(video_comments_features_filename, mode='rb'))
            video_comments_features = self.get_ordered_input_features(video_comments_features)
        return list(video_comments_features.values())
