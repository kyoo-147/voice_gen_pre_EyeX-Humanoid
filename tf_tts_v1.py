# Extra module gne voice
# Name: FastMel
import numpy as np
import soundfile as sf
import yaml
import tensorflow as tf
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

# khởi tạo mô hình fastspeech2 
# https://huggingface.co/tensorspeech
fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")


# khởi tạo mô hình mb_melgan 
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")

# suy luận
processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

# input_ids = processor.text_to_sequence("Recent research at Harvard has shown meditating for as little as 8 weeks, can actually increase the grey matter in the parts of the brain responsible for emotional regulation, and learning.")
input_ids = processor.text_to_sequence("hello Optimus. I'm EyeX AI! Why you don't talk to me ?")

# suy luận fastspeech
mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
    input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
    speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    f0_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
    energy_ratios =tf.convert_to_tensor([1.0], dtype=tf.float32),
)

# suy luận mel
audio_before = mb_melgan.inference(mel_before)[0, :, 0]
audio_after = mb_melgan.inference(mel_after)[0, :, 0]

# lưu lại gen voice
sf.write('./main_au_be.wav', audio_before, 22050, "PCM_16")
sf.write('./main_au_af.wav', audio_after, 22050, "PCM_16")
