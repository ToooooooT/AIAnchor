from flask import Flask, render_template, request, send_file, redirect, url_for
from utils import mkvideo, Content2EmotionalVideo, combineVideo
import os

app = Flask(__name__)

def tts_rvc(text, model_path, input_path, index_path, output_path, tts_wav_file):
    language = "cn"
    os.system(f"python ./TTS/tts.py --text {text} --output_file {tts_wav_file} --language {language}")

    # output voice conversion wav
    speaker_id = 0
    transposition = 0
    f0_method = "rmvpe"
    crepe_hop_length = 160
    feature_index_ratio = 0.78
    voiceless_consonant_protection = 0.33
    command = "python infer-web.py --simple_cli infer"
    command += f" --model_file_name {model_path}"
    command += f" --source_audio_path {input_path}"
    command += f" --output_file_name {output_path}"
    command += f" --feature_index_path {index_path}"
    command += f" --speaker_id {speaker_id}"
    command += f" --transposition {transposition}"
    command += f" --infer_f0_method {f0_method}"
    command += f" --crepe_hop_length {crepe_hop_length}"
    # command += f" --post_resample_rate {args.post_resample_rate}"
    # command += f" --mix_volume_envelope {args.mix_volume_envelope}"
    command += f" --feature_index_ratio {feature_index_ratio}"
    command += f" --voiceless_consonant_protection {voiceless_consonant_protection}"
    # command += f" --formant_shift {args.formant_shift}"
    # command += f" --formant_quefrency {args.formant_quefrency}"
    # command += f" --formant_timbre {args.formant_timbre}"
    os.system("cd TTS/Mangio-RVC-Fork-Simple-CLI && " + command)


def gen_image(text, model_path, output_path):
    os.system(f'python ./TextClassification/img_test.py --text {text} --ckpt {model_path} --save_path {output_path}') 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        output_img_folder = request.form['ImgCache']
        output_video_folder = request.form['VideoFolder']
        who = request.form['char']

        if not os.path.exists(output_img_folder):
            output_img_folder = '/home/toooot/ETtoday/TextClassification/best_img'
        if not os.path.exists(output_video_folder):
            output_video_folder = './output_video/'

        tts_wav_file = "/home/toooot/ETtoday/TTS/Mangio-RVC-Fork-Simple-CLI/saudio/google_voice.wav"
        sentences = [title] + [content]
        rvc_model_path = "Harris_best.pth"
        input_voice_path = "saudio/google_voice.wav"
        index_path = "logs/Harris_best/added_IVF3057_Flat_nprobe_1_Harris_v2.index"
        tc_model_path = "/home/toooot/ETtoday/TextClassification/weights/model_1e-05_64.pth"
        for i, text in enumerate(sentences):
            output_voice_path = f"{i}.wav"
            output_img_path = os.path.join(output_img_folder, f"{i}.jpg")
            tts_rvc(text, rvc_model_path, input_voice_path, index_path, output_voice_path, tts_wav_file)
            gen_image(text, tc_model_path, output_img_path)

        # combine multiple waves and pictures to one video
        mkvideo(image_folder=output_img_folder, 
                audio_folder='./TTS/Mangio-RVC-Fork-Simple-CLI/audio-outputs',
                output_folder=output_video_folder)

        character_video_path = os.path.join(output_video_folder, f"{who}.mp4")
        Content2EmotionalVideo(sentences=sentences, 
                               save_path=character_video_path, 
                               audio_folder='./TTS/Mangio-RVC-Fork-Simple-CLI/audio-outputs',
                               emo_video_folder='./statics',
                               who=who)

        combineVideo(newsVideo_path=os.path.join(output_video_folder, 'final.mp4'),
                     characterVideo_path=character_video_path,
                     save_path=os.path.join(output_video_folder, 'news.mp4'))

        return redirect(url_for('video'))
    return render_template('form.html')


@app.route('/video')
def video():
    # Provide the correct path to your video file
    video_path = '/home/toooot/ETtoday/output_video/news.mp4'
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(port=8787)
