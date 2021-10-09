import pandas as pd
import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 


def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    return " ".join(filtered_words)


print('Video snippets')
with open('groundtruth_dataset.json', 'r') as f:
    with open('video_snippet_train_data.txt', 'w') as output:
        for line in f:
            data = json.loads(line)
            sentence = preprocess(data['snippet']['description'])
            output.write(
                sentence + '\n'
            )

print('Comments')
comments = pd.read_pickle('comments.p')
with open('video_comments_train_data.txt', 'w') as output:
    for comment in comments['text_original'].sample(100000):
        sentence = preprocess(comment)
        output.write(sentence + '\n')

print('Video transcripts')
with open('groundtruth_videos_transcripts.json', 'r') as f:
    with open('video_transcript_train_data.txt', 'w') as output:
        for line in f:
            data = json.loads(line)
            for sentence in data['captions']:
                sentence = preprocess(sentence)
                output.write(sentence + '\n')

print('Video tags')
with open('groundtruth_dataset.json', 'r') as f:
    with open('video_tags_train_data.txt', 'w') as output:
        for line in f:
            data = json.loads(line)
            tags = data['snippet']['tags']
            if len(tags) > 0:
                output.write(
                    ' '.join(tags) + '\n'
                )
