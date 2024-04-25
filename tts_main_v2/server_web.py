import gradio as gr
import librosa
import numpy as np
import torch

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


speaker_embeddings = {
    "BDL": "speaker/cmu_us_bdl_arctic-wav-arctic_a0009.npy",
    "CLB": "speaker/cmu_us_clb_arctic-wav-arctic_a0144.npy",
    "KSP": "speaker/cmu_us_ksp_arctic-wav-arctic_b0087.npy",
    "RMS": "speaker/cmu_us_rms_arctic-wav-arctic_b0353.npy",
    "SLT": "speaker/cmu_us_slt_arctic-wav-arctic_a0508.npy",
}


def predict(text, speaker):
    if len(text.strip()) == 0:
        return (16000, np.zeros(0).astype(np.int16))

    inputs = processor(text=text, return_tensors="pt")

    # limit input length
    input_ids = inputs["input_ids"]
    input_ids = input_ids[..., :model.config.max_text_positions]

    if speaker == "Surprise Me!":
        # load one of the provided speaker embeddings at random
        idx = np.random.randint(len(speaker_embeddings))
        key = list(speaker_embeddings.keys())[idx]
        speaker_embedding = np.load(speaker_embeddings[key])

        # randomly shuffle the elements
        np.random.shuffle(speaker_embedding)

        # randomly flip half the values
        x = (np.random.rand(512) >= 0.5) * 1.0
        x[x == 0] = -1.0
        speaker_embedding *= x

        # speaker_embedding = np.random.rand(512).astype(np.float32) * 0.3 - 0.15
    else:
        speaker_embedding = np.load(speaker_embeddings[speaker[:3]], allow_pickle=True)

    speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)

    speech = model.generate_speech(input_ids, speaker_embedding, vocoder=vocoder)

    speech = (speech.numpy() * 32767).astype(np.int16)
    
    return (16000, speech)


title = "Text-to-Speech based on SpeechT5"

description = """
The <b>SpeechT5</b> model is pre-trained on text as well as speech inputs, with targets that are also a mix of text and speech.
By pre-training on text and speech at the same time, it learns unified representations for both, resulting in improved modeling capabilities.

This space demonstrates the <b>text-to-speech</b> (TTS) checkpoint for the English language.

<b>How to use:</b> Enter some English text and choose a speaker. The output is a mel spectrogram, which is converted to a mono 16 kHz waveform by the HiFi-GAN vocoder. Because the model always applies random dropout, each attempt will give slightly different results.
The <em>Surprise Me!</em> option creates a completely randomized speaker.
"""

article = """
<div style='margin:20px auto;'>

<p>References: <a href="https://arxiv.org/abs/2110.07205">SpeechT5 paper</a> |
<a href="https://github.com/microsoft/SpeechT5/">original GitHub</a> |
<a href="https://huggingface.co/mechanicalsea/speecht5-tts">original weights</a></p>

<pre>
@article{Ao2021SpeechT5,
  title   = {SpeechT5: Unified-Modal Encoder-Decoder Pre-training for Spoken Language Processing},
  author  = {Junyi Ao and Rui Wang and Long Zhou and Chengyi Wang and Shuo Ren and Yu Wu and Shujie Liu and Tom Ko and Qing Li and Yu Zhang and Zhihua Wei and Yao Qian and Jinyu Li and Furu Wei},
  eprint={2110.07205},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  year={2021}
}
</pre>

<p>Speaker embeddings were generated from <a href="http://www.festvox.org/cmu_arctic/">CMU ARCTIC</a> using <a href="https://huggingface.co/mechanicalsea/speecht5-vc/blob/main/manifest/utils/prep_cmu_arctic_spkemb.py">this script</a>.</p>

</div>
"""

examples = [
    ["As a Data Scientist, I'll be demonstrating my speaking voice in this example. If you don't like my voice, you can choose a different one by setting the speaker parameter.", "BDL (male)"],
    ["The octopus and Oliver went to the opera in October.", "CLB (female)"],
    ["She sells seashells by the seashore. I saw a kitten eating chicken in the kitchen.", "RMS (male)"],
    ["Brisk brave brigadiers brandished broad bright blades, blunderbusses, and bludgeons—balancing them badly.", "SLT (female)"],
    ["A synonym for cinnamon is a cinnamon synonym.", "BDL (male)"],
    ["How much wood would a woodchuck chuck if a woodchuck could chuck wood? He would chuck, he would, as much as he could, and chuck as much wood as a woodchuck would if a woodchuck could chuck wood.", "CLB (female)"],
]

gr.Interface(
    fn=predict,
    inputs=[
        gr.Text(label="Input Text"),
        gr.Radio(label="Speaker", choices=[
            "BDL (male)",
            "CLB (female)",
            "KSP (male)",
            "RMS (male)",
            "SLT (female)",
            "Surprise Me!"
        ],
        value="BDL (male)"),
    ],
    outputs=[
        gr.Audio(label="Generated Speech", type="numpy"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
).launch()
