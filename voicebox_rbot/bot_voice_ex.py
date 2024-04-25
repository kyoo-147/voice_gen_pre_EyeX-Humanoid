# Example: Use gTTS with a vocoder effect to speak in a robotic voice

from voicebox import SimpleVoicebox
from voicebox.tts import gTTS
from voicebox.effects import Vocoder, Normalize

voicebox = SimpleVoicebox(
    tts=gTTS(),
    effects=[Vocoder.build(), Normalize()],
)

voicebox.say('Hello, world! How are you today?')