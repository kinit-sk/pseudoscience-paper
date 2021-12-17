from os import listdir
import json
from pseudoscientificvideosdetection.PseudoscienceClassifier import PseudoscienceClassifier

video_files = [f for f in listdir('videosdata/videos') if f.endswith('.json')]
pseudoscienceClassifier = PseudoscienceClassifier()

with open('predictions.txt', 'w') as f:
    f.write('')

for file_name in video_files:
    with open('videosdata/videos/' + file_name, 'r') as f:
        video = json.load(f)
    if not video: continue

    prediction = pseudoscienceClassifier.classify(video_details=video)
    print(prediction)
    with open('predictions.txt', 'a') as f:
        f.write(json.dumps({
            'video': video,
            'prediction': prediction
        }))
