from txtai.pipeline import TextToSpeech

# Create text-to-speech model
tts = TextToSpeech()

import librosa.display
import matplotlib.pyplot as plt

text = "Hello User! I'm Rabbit AI, How can i help you today ?"

# Generate raw waveform speech
speech, rate = tts(text), 22050

# Print waveplot
plt.figure(figsize=(15, 5))
plot = librosa.display.waveshow(speech, sr=rate)

import soundfile as sf

def play(speech):
  # Convert to MP3 to save space
  sf.write("speech.wav", speech, 22050)
  
play(speech)


# Beginning of The Great Gatsby from Project Gutenberg
# https://www.gutenberg.org/ebooks/64317

text = """
In my younger and more vulnerable years my father gave me some advice
that I've been turning over in my mind ever since.

“Whenever you feel like criticizing anyone,” he told me, “just
remember that all the people in this world haven't had the advantages
that you've had.”

He didn't say any more, but we've always been unusually communicative
in a reserved way, and I understood that he meant a great deal more
than that.
"""

speech = tts(text)
play(speech)