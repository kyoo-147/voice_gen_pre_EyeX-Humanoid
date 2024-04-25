# Unfortunately, it's not currently possible to use these libraries without 
# installing a large number of dependencies.
from txtai.pipeline import TextToSpeech
# import os
import soundfile as sf

tts = TextToSpeech()

text = "Text To Speech models have made great strides in quality over the last few years."
# Generate raw waveform speech
speech, rate = tts(text), 22050
# 
def play(speech):
  # Convert to MP3 to save space
  sf.write("speech1.wav", speech, 22050)
#   !ffmpeg -i speech.wav -y -b:a 64 speech.mp3 2> /dev/null
  
# text = """
# Hello User, I'm Rabbit, How can i help you today ?
# """
sf.write("speech1.wav", speech, 22050)

# speech = tts(text)
# play(speech)

# Phiên dịch
# from txtai.pipeline import Transcription
# # Transcribe files
# transcribe = Transcription("openai/whisper-base")
# # Print result
# transcribe(speech, rate)