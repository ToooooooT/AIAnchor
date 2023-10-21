from moviepy.editor import VideoFileClip, CompositeVideoClip, vfx
from moviepy.editor import VideoFileClip, CompositeVideoClip


def combineVideo(newsVideo_path, characterVideo_path, save_path):
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
    final_video.write_videofile(save_path, codec='libx264')
