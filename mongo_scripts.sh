# run mongo on WSL
sudo /usr/bin/mongod --fork --config /etc/mongod.conf

# import data
mongoimport -d youtube_pseudoscience_dataset -c groundtruth_videos --file=yaudit-data/groundtruth_dataset.json --jsonArray
mongoimport -d youtube_pseudoscience_dataset -c groundtruth_videos_transcripts --file=yaudit-data/groundtruth_videos_transcripts.json --jsonArray