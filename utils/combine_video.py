from moviepy.editor import VideoFileClip, CompositeVideoClip, vfx
from moviepy.video.tools.subtitles import SubtitlesClip, TextClip
import wave
import os


def add_subtitles_clip(video_clip, sentences, repeat_duration):
    subtitles = []
    total_sec = float(0)
    for i in range(len(sentences)):
        txt = TextClip(sentences[i], 
                       font='/home/toooot/ETtoday/NotoSansTC-VariableFont_wght.ttf', 
                       align='center', 
                       fontsize=20,
                       bg_color='white',
                       stroke_width=10) \
                        .set_duration(repeat_duration[i]) \
                        .set_start(total_sec) \
                        .set_position(('center', 'bottom'))
        subtitles.append(txt)
        total_sec += repeat_duration[i]

    return CompositeVideoClip([video_clip, *subtitles])


def get_audio_length(wav_file):
    with wave.open(wav_file, 'rb') as audio_file:
        sample_rate = audio_file.getframerate()
        total_frames = audio_file.getnframes()
        audio_length = total_frames / float(sample_rate)
    return audio_length


def audio_length(audio_folder):
    audio_files = os.listdir(audio_folder)
    audio_files = sorted(audio_files, key=lambda x : int(x.split('.')[0]))
    audio_lengthes = []
    for p in audio_files:
        audio_lengthes.append(get_audio_length(f'{audio_folder}/{p}'))
    return audio_lengthes


def combineVideo(newsVideo_path, characterVideo_path, save_path, sentences, audio_folder):
    '''
    Args:
        newsVideo_path: output video path from mk_video function
        characterVideo_path: output video from Content2EmotionalVideo function
    '''
    video1 = VideoFileClip(newsVideo_path)
    video2 = VideoFileClip(characterVideo_path)
    video2 = video2.resize(height=130, width=130)
    video2 = video2.fx(vfx.mask_color, color=(0, 255, 0), thr=179, s=19.9)
    video2 = video2.set_position(('right', 'bottom'))
    final_video = CompositeVideoClip([video1, video2])
    audio_lengthes = audio_length(audio_folder)
    sentences, audio_lengthes = resegment_subtitle(sentences, audio_lengthes)
    final_video = add_subtitles_clip(final_video, sentences, audio_lengthes)
    final_video.write_videofile(save_path, codec='libx264')

def resegment_subtitle(subtitles, time):
    new_subtitles = []
    new_time = []

    for subtitle, t in zip(subtitles, time):
        subtitle_length = len(subtitle)

        segment_length = 10

        for i in range(0, subtitle_length, segment_length):
            new_subtitle_segment = subtitle[i:i + segment_length]
            new_time_segment = t * len(new_subtitle_segment) / subtitle_length
            new_subtitles.append(new_subtitle_segment)
            new_time.append(new_time_segment)
    return new_subtitles, new_time
