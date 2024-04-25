# Use eSpeak NG at 120 WPM and en-us voice as the TTS engine
from voicebox import reliable_tts
from voicebox.tts import ESpeakConfig, ESpeakNG, gTTS

# Wrap multiple TTSs in retries and caches
tts = reliable_tts(
    ttss=[
        # Prefer using online TTS first
        gTTS(),
        # Fall back to offline TTS if online TTS fails
        ESpeakNG(ESpeakConfig(speed=100, voice='en-us')),
    ],
)

# Add some voice effects
from voicebox.effects import Vocoder, Glitch, Normalize

effects = [
    Vocoder.build(),    # Make a robotic, monotone voice
    Glitch(),           # Randomly repeat small sections of audio
    Normalize(),        # Remove DC and make volume consistent
]

# Build audio sink
from voicebox.sinks import Distributor, SoundDevice, WaveFile

sink = Distributor([
    SoundDevice(),          # Send audio to playback device
    WaveFile('speech.wav'), # Save audio to speech.wav file
])

# Build the voicebox
from voicebox import ParallelVoicebox
from voicebox.voiceboxes.splitter import SimpleSentenceSplitter

# Parallel voicebox doesn't block the main thread
voicebox = ParallelVoicebox(
    tts,
    effects,
    sink,
    # Split text into sentences to reduce time to first speech
    text_splitter=SimpleSentenceSplitter(),
)

# Speak!
voicebox.say('Hello Optimus! I am Rabbit! Why you dont talk to me?')

# Wait for all audio to finish playing before exiting
voicebox.wait_until_done()