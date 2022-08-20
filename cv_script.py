from dataset.DatasetUtils import DatasetUtils
from classifier.training.ClassifierTraining import ClassifierTraining

# Create Objects
dataset = DatasetUtils()

# Create ClassifierTraining Object
classifierTrainingObject = ClassifierTraining(dataset_object=dataset)

classifierTrainingObject.train_model()
