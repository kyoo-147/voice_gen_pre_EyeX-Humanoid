# import gradio as gr
# import librosa
import numpy as np
import torch

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

from scipy.io.wavfile import write

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


speaker_embeddings = {
    "CLB": "speaker/cmu_us_clb_arctic-wav-arctic_a0144.npy"
}


def predict(text, speaker):
    if len(text.strip()) == 0:
        return (16000, np.zeros(0).astype(np.int16))

    inputs = processor(text=text, return_tensors="pt")

    # limit input length
    input_ids = inputs["input_ids"]

    input_ids = input_ids[..., :model.config.max_text_positions]

    speaker_embedding = np.load(speaker_embeddings[speaker[:3]], allow_pickle=True)

    speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)

    speech = model.generate_speech(input_ids, speaker_embedding, vocoder=vocoder)

    speech = (speech.numpy() * 32767).astype(np.int16)
    return (16000, speech)

text = "Hello, User! I'm Rabbit AI. How can i help you today ?"
speaker = "CLB"

results = predict(text, speaker)
print(results)

sample_rate = 16000
# write("output_audio.wav", sample_rate, results)
write("output_audio1.wav", results[0], results[1])


# outputs=[gr.Audio(label="Generated Speech", type="numpy")]
