import streamlit as st

st.title('Music Exploration Tool')

import requests
import json
import shutil
from tqdm import tqdm
import numpy as np
import faiss

from urllib.parse import quote

output_dir = "likes_dump"

likes_dump = json.loads(open('likes_dump.json').read())

api_key = "2d2a98aa352c1eeb48aff96c34903dc4"
shared_secret = "ee336f501f70354f703358ad142cbe8d"

embeddings = np.load("embeddings.npy")
index = faiss.read_index("likes_index.faiss")

def query_lastfm(artist_name, track_name):
    res = requests.get(url = f"https://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={api_key}&artist={artist_name}&track={track_name}&format=json")
    return res.json()

if st.button("Load 16 Random Songs"):
    song_idx = np.random.choice(len(likes_dump), 16)
    songs = [likes_dump[i] for i in song_idx]
    for idx, song in enumerate(songs):
        col1, col2 = st.columns(2)
        metadata = query_lastfm(song['artist_name'], song['track_name'])
        with col1:
            st.write("**Track:**", song['track_name'])
            st.write("**Artist:**", song['artist_name'])
            st.write("**Album:**", song['album_name'])
            print(embeddings[song_idx[idx]])
            print(embeddings[song_idx[idx]].shape)
            D, I = index.search(embeddings[song_idx[idx]].reshape(1,-1), 5)
            
        with col2:
            try:
                st.image(metadata["track"]["album"]["image"][-1]["#text"])
            except Exception as e:
                print(e)
                st.write("No album art found")
        st.audio(song['preview_url'])
        st.write("**Similar Tracks:**")
        for d, i in zip(D[0], I[0]):
            col3, col4 = st.columns(2)
            if i != song_idx[idx]:
                sim_metadata = query_lastfm(likes_dump[int(i)]['artist_name'], likes_dump[int(i)]['track_name'])
                with col3:
                    st.write(f"{likes_dump[int(i)]['track_name']} by {likes_dump[int(i)]['artist_name']}")
                with col4:
                    try:
                        st.image(sim_metadata["track"]["album"]["image"][-1]["#text"])
                    except Exception as e:
                        print(e)
                        st.write("No album art found")
                st.audio(likes_dump[int(i)]['preview_url'])
        st.write('---')