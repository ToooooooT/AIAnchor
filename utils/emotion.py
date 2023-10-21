from KeyMojiAPI import KeyMoji
import re
from moviepy.editor import VideoFileClip, concatenate_videoclips
import wave
import os


def get_audio_length(wav_file):
    with wave.open(wav_file, 'rb') as audio_file:
        sample_rate = audio_file.getframerate()
        total_frames = audio_file.getnframes()
        audio_length = total_frames / float(sample_rate)
    return audio_length


def split_text(text):
    pattern = r'[，。]'
    segments = re.split(pattern, text)
    segments = [segment.strip() for segment in segments if segment.strip()]
    return segments


def Text2Emotion(text):
    keymoji = KeyMoji(username="", keymojiKey="")
    sense2Result = keymoji.sense2(text, model="general", userDefinedDICT={
        "positive": [], "negative": [], "cursing": []})
    if sense2Result['results'][0]['sentiment'] == 'positive':
        return 1
    elif sense2Result['results'][0]['sentiment'] == 'negative':
        return -1
    else:
        return 0


def MakeEmotionalVideo(emo_video_list, repeat_durations, save_path):
    video_clips = []
    reverse = False
    for i, path in enumerate(emo_video_list):
        videoclip = VideoFileClip(path)
        dur = videoclip.duration
        for _ in range(int(repeat_durations[i]/dur)):
            video_clips.append(videoclip)
        video_clips.append(videoclip.subclip(
            0, repeat_durations[i]-dur*int(repeat_durations[i]/dur)))

    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile(save_path, codec='libx264')


def extract_number(filename):
    return int(filename.split('.')[0])


def Content2EmotionalVideo(sentences: list, save_path, audio_folder, emo_video_folder, who):
    '''
    Args:
        sentences: list of text
        save_path: emotinal video save path
        audio_folder: wav folder from rvc
        emo_video_folder: emotional video folder
        who: character
    '''
    # sentences = split_text(content)
    emo_video_list = []
    repeat_durations = []
    for sentence in sentences:
        emo = Text2Emotion(sentence)
        if emo == 1:
            emo_video_list.append(os.path.join(emo_video_folder, who, 'happy.mp4'))
        elif emo == -1:
            emo_video_list.append(os.path.join(emo_video_folder, who, 'sad.mp4'))
        else:
            emo_video_list.append(os.path.join(emo_video_folder, who, 'neutral.mp4'))
    audio_files = os.listdir(audio_folder)
    audio_files = sorted(audio_files, key=extract_number)
    for audio_file in audio_files:
        repeat_durations.append(get_audio_length(
            f'{audio_folder}/{audio_file}'))
    MakeEmotionalVideo(emo_video_list, repeat_durations, save_path)
