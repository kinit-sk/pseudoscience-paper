import nltk
import json
import pandas as pd


def transform_video(video):
    return {
        "annotation": {
            "annotations": [
            ],
            "label": video['annotation'],
            "manual_review_label": video['annotation']
        },
        "etag": "",
        "id": video['uuid'],
        "isSeed": True,
        "kind": "youtube#video",
        "relatedVideos": [
        ],
        "search_term": "",
        "snippet": {
            "categoryId": video['category_id'],
            "channelId": video['channel_id'],
            "channelTitle": "",
            "defaultAudioLanguage": video['default_audio_language'],
            "description": video['description'],
            "liveBroadcastContent": "none",
            "localized": {
                "description": video['description'],
                "title": video['title']
            },
            "publishedAt": video['published_at'].isoformat(),
            "tags": [
            ],
            "thumbnails": {
            },
            "title": video['title']
        },
        "statistics": {
            "commentCount": video['comment_count'],
            "dislikeCount": video['dislike_count'],
            "favoriteCount": video['favourite_count'],
            "likeCount": video['like_count'],
            "viewCount": video['view_count']
        }
    }


def transform_video_to_transcript(video):
    return {
        "captions": [
            sent.lower()
            for line in video['clean_transcript']
            for sent in nltk.sent_tokenize(line)
        ],
        "id": video['uuid']
    }


videos = pd.read_pickle('videos.p')

with open('groundtruth_dataset.json', 'w') as f:
    for _, video in videos.iterrows():
        data = json.dumps(transform_video(video))
        f.write(f'{data}\n')


with open('groundtruth_videos_transcripts.json', 'w') as f:
    for _, video in videos.iterrows():
        data = json.dumps(transform_video_to_transcript(video))
        f.write(data + '\n')
