from google.cloud import texttospeech
import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--text', default=None, type=str, help='text for TTS')
    parse.add_argument('--file', default=None, type=str, help='file containing text for TTS')
    parse.add_argument('--language', default="cn", type=str, help='language')
    parse.add_argument('--output_file', default='/home/toooot/ETtoday/TTS/wav/google_voice.wav', type=str, help='output file path of wav')
    args = parse.parse_args()
    return args


def text_to_speech(text, output_file, language):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    if language == "cn":
        voice = texttospeech.VoiceSelectionParams(
            language_code="cmn-CN",  # Chinese (Mandarin)
            name="cmn-CN-Wavenet-B",  # Choose a Mandarin voice model
        )
    elif language == "jp":
        voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",  # Japanese
            name="ja-JP-Wavenet-B",  # Choose a Japanese voice model
        )
    elif language == "en":
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",  # English (United States)
            name="en-US-Wavenet-D",  # Choose an English voice model
        )
    else:
        raise ValueError # not implement language

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)


if __name__ == '__main__':
    args = parse_args()
    text = None
    if args.text != None:
        text = args.text
    if args.file != None:
        with open(args.file, 'r') as f:
            text = f.read()
    if text != None:
        text_to_speech(text, args.output_file, args.language)
    else:
        print("No text for TTS!")

