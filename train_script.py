import os

from dataset.DatasetUtils import DatasetUtils
from classifier.featureengineering.FeatureEngineeringModels import FeatureEngineeringModels
from classifier.training.ClassifierTraining import ClassifierTraining

# Create Objects
dataset = DatasetUtils()
featureEngineeringModels = FeatureEngineeringModels(dataset_object=dataset)

# # Video Snippet
# # Generate Video Snippet fastText input features
# featureEngineeringModels.prepare_fasttext_data(model_type='video_snippet', overwrite=True)
# # Fine-tune a fastText model for Video Snippet
# featureEngineeringModels.finetune_model(model_type='video_snippet', overwrite=True)

# # # Video Tags
# # # Generate Video Tags fastText input features
# # featureEngineeringModels.prepare_fasttext_data(model_type='video_tags')
# # # Fine-tune a fastText model for Video Tags
# # featureEngineeringModels.finetune_model(model_type='video_tags')

# # Video Transcript
# # Generate Video Transcript fastText input features
# featureEngineeringModels.prepare_fasttext_data(model_type='video_transcript', overwrite=True)
# # Fine-tune a fastText model for Video Transcript
# featureEngineeringModels.finetune_model(model_type='video_transcript', overwrite=True)

# # Video Comments
# # Generate Video Comments fastText input features
# featureEngineeringModels.prepare_fasttext_data(model_type='video_comments', overwrite=True)
# # Fine-tune a fastText model for Video Comments
# featureEngineeringModels.finetune_model(model_type='video_comments', overwrite=True)

# Create ClassifierTraining Object
classifierTrainingObject = ClassifierTraining(dataset_object=dataset)

classifierTrainingObject.train_model_no_cv()
