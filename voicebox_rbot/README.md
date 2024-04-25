# Tạo âm thanh bot với voicebox

Thư viện T2S với xây dựng và hỗ trợ cho đa luồng âm thanh 


```python
# Chúng tôi ví dụ cung cấp gTTS với hiệu ứng vocoder thành âm thanh robotic

from voicebox import SimpleVoicebox
from voicebox.tts import gTTS
from voicebox.effects import Vocoder, Normalize

voicebox = SimpleVoicebox(
    tts=gTTS(),
    effects=[Vocoder.build(), Normalize()],
)

voicebox.say('Hello, world! How are you today?')
```

## Thiết lập cài đặt

1. `pip install voicebox-tts`
2. `sudo apt install libportaudio2`

### gTTS [🌐](https://github.com/pndurette/gTTS)

Online TTS engine used by Google Translate.

- Class: [`voicebox.tts.gTTS`](voicebox.tts.gtts.gTTS)
- Thiết lập:
  1. `pip install "voicebox-tts[gtts]"`
  2. Cài đặt ffmpeg hoặc libav cho `pydub` ([docs](https://github.com/jiaaro/pydub#getting-ffmpeg-set-up))

### Pico TTS

TTS ngoại tuyến cơ bản

- Class: [`voicebox.tts.PicoTTS`](voicebox.tts.picotts.PicoTTS)
- Thiết lập:
  - Trên Debian/Ubuntu: `sudo apt install libttspico-utils`

## Hiệu ứng
Bạn có thể thêm các hiệu ứng âm thanh tùy thichs tại repo gốc của nhà sản xuất
Tuy nhiên việc đó không cần thiết trong truờng hợp này

```python
from voicebox import SimpleVoicebox
from voicebox.effects import PedalboardEffect
import pedalboard

voicebox = SimpleVoicebox(
    effects=[
        PedalboardEffect(pedalboard.Reverb()),
        ...,
    ]
)
```

## Ví dụ

### Minimal

```python
# "Hello, world!"
from voicebox import SimpleVoicebox

voicebox = SimpleVoicebox()
voicebox.say('Hello, world!')
```

### Nâng cao

```python
from voicebox import reliable_tts
from voicebox.tts import ESpeakConfig, ESpeakNG, gTTS

tts = reliable_tts(
    ttss=[
        gTTS(),
        ESpeakNG(ESpeakConfig(speed=120, voice='en-us')),
    ],
)

from voicebox.effects import Vocoder, Glitch, Normalize

effects = [
    Vocoder.build(),    # nghe giống như robot, monotone 
    Glitch(),           # Lặp lại ngẫu nhiên các phần nhỏ của âm thanh
    Normalize(),        # Loại bỏ DC và làm cho âm lượng nhất quán
]

# 
from voicebox.sinks import Distributor, SoundDevice, WaveFile

sink = Distributor([
    SoundDevice(),          # phát âm thanh
    WaveFile('speech.wav'), # 
])

from voicebox import ParallelVoicebox
from voicebox.voiceboxes.splitter import SimpleSentenceSplitter

voicebox = ParallelVoicebox(
    tts,
    effects,
    sink,
    # Chia văn bản thành câu để giảm thời gian cho lời nói đầu tiên
    text_splitter=SimpleSentenceSplitter(),
)

# !
voicebox.say('Hello, world!')

voicebox.wait_until_done()
```