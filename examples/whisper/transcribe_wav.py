import os
import sys

from .whisper import decode_wav_file


def main():
    if len(sys.argv) < 2:
        wav_file = os.path.join(os.path.dirname(__file__), 'assets', 'ever_tried.wav')
        model = 'tiny.en'
        src_lang = 'en'
    else:
        wav_file = sys.argv[1]
        model = sys.argv[2]
        src_lang = sys.argv[3]
    text = decode_wav_file(wav_file,model,task='transcribe',src_lang=src_lang)
    print(text)


if __name__ == '__main__':
    main()
