from time import sleep
from youtubehelpers.YouTubeVideoDownloader import YouTubeVideoDownloader
import json
import pandas as pd
from os.path import exists

recommendations = pd.read_pickle('annotated-home-page-results.pickle')
recommendations = recommendations.loc[recommendations['annotation_label'] == 'not annotated']
youtube_ids = set(recommendations['youtube_id'])
print(len(youtube_ids))

ytDownloader = YouTubeVideoDownloader()

for video_id in youtube_ids:
    path = f'videosdata/videos/{video_id}.json'
    if not exists(path):
        print('downloading video', video_id)
        video_details = ytDownloader.download_video(video_id=video_id)
        with open(path, 'w') as f:
            json.dump(video_details, f)
        print('done')
        sleep(10)
