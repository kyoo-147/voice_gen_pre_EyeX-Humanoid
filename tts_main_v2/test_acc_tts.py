from txtai.pipeline import Transcription

transcribe = Transcription("openai/whisper-base")

speech = "out.wav"

rate = 22050

results = transcribe(speech, rate)
 
print("Transcription: ", results)