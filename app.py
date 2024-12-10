import base64
import os
import shutil
import tempfile
from enum import Enum

import cv2
import streamlit as st
from openai import OpenAI
from yt_dlp import YoutubeDL

# OpenAI APIキー
with open("key.txt", "r") as f:
    OPENAI_API_KEY = f.read()
client = OpenAI(api_key=OPENAI_API_KEY)

# YouTube動画をダウンロードする関数（yt-dlpを使用）
def download_video_yt_dlp(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': tempfile.mkdtemp() + '/%(id)s.%(ext)s',
        'quiet': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_path = ydl.prepare_filename(info_dict)
            return video_path, os.path.dirname(video_path)
    except Exception as e:
        st.error(f"動画のダウンロード中にエラーが発生しました: {e}")
        return None, None

# 動画からフレームを抽出する関数
def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    success = True
    while success:
        success, frame = cap.read()
        if success and frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

# フレームの画像をbase64エンコードする関数
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# GPT-4Vでフレームを解析する関数
def describe_frame_with_gpt4v(frame, level):
    # フレームを一時ファイルに保存
    frame_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(frame_path, frame)
    base64_img = encode_image(frame_path)

    level_text = "私はサッカーの初心者です。" if level.value == 1 else "私はサッカーに詳しいです。"
    
    try:
        # GPT-4Vに画像を送信して解説を取得
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                { "role": "system", "content": "あなたはサッカー解説者です。"},
                { "role": "user", 
                    "content": [
                        {
                        "type": "text",
                        "text": f"{level_text} この画像から試合状況を詳細に解説してください。選手の入場シーンなど、試合内容と関係ないところは簡潔に説明してください。",
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/jpeg;base64,{base64_img}"
                            },
                        },
                    ]}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"シーン説明の生成中にエラーが発生しました: {e}")
        return "説明を生成できませんでした。"
    finally:
        os.remove(frame_path)

# Streamlitアプリ
st.title("サッカー動画解説 (gpt-4o-mini)")

class Level(Enum):
    def __str__(cls):
        return cls.name

    初心者 = 0
    詳しい = 1


youtube_url = st.text_input("YouTubeのURLを入力してください:")
frame_interval = st.number_input("フレーム間隔（フレーム数ごとに抽出）", value=30, step=1)
level = st.radio("あなたはサッカーに詳しいですか？", Level, horizontal=False)
st.write(f'{level}を選択しました')

if youtube_url:
    st.write("動画をダウンロードしています...")
    video_path, temp_dir = download_video_yt_dlp(youtube_url)
    
    if video_path:
        st.write("動画をダウンロードしました。フレームを抽出中...")
        
        frames = extract_frames(video_path, frame_interval)
        st.write(f"{len(frames)}個のフレームを抽出しました。")
        
        st.write("シーンの説明を生成中...")
        for idx, frame in enumerate(frames):
            st.write(f"フレーム {idx + 1} の説明を生成中...")
            description = describe_frame_with_gpt4v(frame, level)
            st.image(frame, caption=description, use_column_width=True)
        
        # 一時ディレクトリを削除
        shutil.rmtree(temp_dir)
    else:
        st.error("動画のダウンロードに失敗しました。URLを確認してください。")

st.write("解析が完了しました。")
