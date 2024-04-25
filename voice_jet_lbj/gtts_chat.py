from gtts import gTTS
import os
from threading import Thread
from playsound import playsound

class TextToSpeech:
    def text_to_speech(self, text):
        tts = gTTS(text=text, lang='en')
        tts.save("output_audio.mp3")
        playsound("output_audio.mp3")
        os.remove("output_audio.mp3")

def get_text_and_convert(tts):
    while True:
        text = input("Enter text (type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        if text:
            print("Converting text to speech...")
            Thread(target=tts.text_to_speech, args=(text,)).start()

def main():
    tts = TextToSpeech()
    get_text_and_convert(tts)

if __name__ == "__main__":
    main()
