import os

from dataset.DatasetUtils import DatasetUtils
from classifier.featureengineering.FeatureEngineeringModels import FeatureEngineeringModels
from classifier.training.ClassifierTraining import ClassifierTraining

# Create Objects
dataset = DatasetUtils()
featureEngineeringModels = FeatureEngineeringModels(dataset_object=dataset)


# Create ClassifierTraining Object
classifierTrainingObject = ClassifierTraining(dataset_object=dataset)

classifierTrainingObject.train_model_no_cv()
