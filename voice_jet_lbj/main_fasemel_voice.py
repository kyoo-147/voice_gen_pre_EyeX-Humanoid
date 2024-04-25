# main module voice cho eyex
# 

import numpy as np
import tensorflow as tf

from tensorflow_tts.inference import TFAutoModel, AutoConfig, AutoProcessor

from tensorflow_tts.models import TFFastSpeech, TFFastSpeech2
from tensorflow_tts.models import TFMelGANGenerator

import soundfile as sf

from pydub import AudioSegment
from pydub.playback import play

def run_melgan(mel_spec, quantization):
    model_name = f'melgan_{quantization}.tflite'
    
    feats = np.expand_dims(mel_spec, 0)
    interpreter = tf.lite.Interpreter(model_path=model_name)
    
    # interpreter = tf.lite.Interpreter(model_path=model_name)

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'],  [1, feats.shape[1], feats.shape[2]], strict=True)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], feats)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output


def run_parallel_wavegan(melspec, quantization):
    model_name = f'parallel_wavegan_{quantization}.tflite'
    feats = np.expand_dims(melspec, 0)
    interpreter = tf.lite.Interpreter(model_path=model_name)

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'],  [1, feats.shape[1], feats.shape[2]], strict=True)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], feats)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output

# Prepare input data.
def fastspeech_prepare_input(input_ids):
  input_ids = tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0)
  return (input_ids,
          tf.convert_to_tensor([0], tf.int32),
          tf.convert_to_tensor([1.0], dtype=tf.float32),
          tf.convert_to_tensor([1.0], dtype=tf.float32),
          tf.convert_to_tensor([1.0], dtype=tf.float32))

# Test the model on random input data.
def fastspeech_infer(tflite_model_path, input_text):
  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  processor = AutoProcessor.from_pretrained(pretrained_path="ljspeech_mapper.json")
  input_ids = processor.text_to_sequence(input_text.lower())
  interpreter.resize_tensor_input(input_details[0]['index'], 
                                  [1, len(input_ids)])
  interpreter.resize_tensor_input(input_details[1]['index'], 
                                  [1])
  interpreter.resize_tensor_input(input_details[2]['index'], 
                                  [1])
  interpreter.resize_tensor_input(input_details[3]['index'], 
                                  [1])
  interpreter.resize_tensor_input(input_details[4]['index'], 
                                  [1])
  interpreter.allocate_tensors()
  input_data = fastspeech_prepare_input(input_ids)
  for i, detail in enumerate(input_details):
    input_shape = detail['shape_signature']
    interpreter.set_tensor(detail['index'], input_data[i])

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  return (interpreter.get_tensor(output_details[0]['index']),
          interpreter.get_tensor(output_details[1]['index']))
  

def run_tts_inference(text, model_name='Tacotron2', vocoder_name='MB-MelGAN', quantization='float16'):
    if model_name == 'FastSpeech2':
        _, tac_output = fastspeech_infer('fastspeech_quant.tflite', text)
        tac_output = np.squeeze(tac_output)
        sample_rate = 22050
    if vocoder_name == 'MelGAN':
        waveform = run_melgan(tac_output, quantization)
        waveform = np.squeeze(waveform)

    sf.write('output.wav', waveform, sample_rate)
    sound = AudioSegment.from_file('output.wav', format='wav')
    play(sound)
    
    print(waveform)
    return waveform, sample_rate

tts_model = 'FastSpeech2' #@param ["Tacotron2", "FastSpeech2", "Glow-TTS"]
vocoder_model = 'MelGAN' #@param ["MelGAN", "MB-MelGAN", "PWGAN"]
quantization = 'float16' #@param ["dr", "float16"]
text = "Hello, User! I am Rabbit. How can i help you today?"

run_tts_inference(text, tts_model, vocoder_model, quantization)

# print(run_tts_inference(text, tts_model, vocoder_model, quantization))



