import pandas as pd
import os
import json


def transform_comment(video_uuid, comment_ids):
    return {
        "comments": comment_ids,
        "id": video_uuid
    }


def comment_to_youtube_response(comment):
    return {
        'snippet': {
            'topLevelComment': {
                'snippet': {
                    'authorDisplayName': comment['author_display_name'],
                    'authorProfileImageUrl': comment['author_profile_image_url'],
                    'authorChannelUrl': comment['author_channel_url'],
                    'textDisplay': comment['text_display'],
                    'textOriginal': comment['text_original'],
                    'parentId': comment['parent_uuid'],
                    'canRate': comment['can_rate'],
                    'likeCount': comment['like_count'],
                    'publishedAt': comment['published_at'].isoformat(),
                    'updatedAt': comment['youtube_update_timestamp'].isoformat(),
                    # 'authorChannelId': {
                    #     'value': comment['author_channel_uuid']
                    # }
                }
            },
            'canReply': comment['can_reply'],
            'totalReplyCount': comment['total_reply_count'],
            'isPublic': comment['is_public']
        }
    }


comments = pd.read_pickle('comments.p')
videos = pd.read_pickle('videos.p')

video_uuids = videos.set_index('id')['uuid'].to_dict()
comments['video_uuid'] = comments['video_id'].apply(video_uuids.get)
comments = comments.loc[~comments['video_uuid'].isna()]

comments_by_video = comments.groupby('video_uuid')['uuid'].apply(list).to_dict()
with open('groundtruth_videos_comments_ids.json', 'w') as f:
    for video_uuid, comment_ids in comments_by_video.items():
        transformed = transform_comment(video_uuid, comment_ids)
        data = json.dumps(transformed)
        f.write(f'{data}\n')

comments_by_video = comments.groupby('video_uuid')
for video_uuid, video_comments in comments_by_video:
    body = '\n'.join([
        json.dumps(comment_to_youtube_response(comment))
        for _, comment in video_comments.iterrows()
    ])

    os.makedirs(
        f'comments/{video_uuid}',
        exist_ok=True
    )

    with open(f'comments/{video_uuid}/{video_uuid}.json', 'w') as f:
        f.write(body)
