from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips, clips_array
import os
import cv2


def extract_number(filename):
    return int(filename.split('.')[0])


def mkvideo(image_folder, audio_folder, output_folder):
    image_files = os.listdir(image_folder)
    image_files = sorted(image_files, key=extract_number)
    final_video_list = []
    for image_file in image_files:
        video_clips = []
        final_video_clips = []
        audio_file = f'{audio_folder}/{image_file.replace(".jpg", ".wav")}'
        image_file = f'{image_folder}/{image_file}'
        image = cv2.imread(image_file)
        resized_image = cv2.resize(image, (480, 270))
        cv2.imwrite(image_file, resized_image)
        print(audio_file)
        audio_clip = AudioFileClip(audio_file)
        image_clip = ImageClip(image_file)
        image_clip = image_clip.resize(width=480, height=270)
        image_clip = image_clip.set_duration(audio_clip.duration)
        video_clip = image_clip.set_audio(audio_clip)
        image_clip = image_clip.set_fps(24)
        video_clips.append(video_clip)
        final_video_clips.append(video_clips)
        final_video_list.append(clips_array([video_clips]))

    final_video = concatenate_videoclips(final_video_list)
    final_video.write_videofile(
        f'{output_folder}/final.mp4', codec='libx264', fps=24)
