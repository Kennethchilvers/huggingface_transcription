import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer
from pydub import AudioSegment
from pydub.silence import split_on_silence
model_checkpoint = "facebook/wav2vec2-base-960h" 
#path to audio file that will be split
sound = AudioSegment.from_wav("Internship_project/Audio/Testdepo.wav")

#Spliting the audio file based on silence parameters
audio_splits = split_on_silence(sound, min_silence_len=600, silence_thresh=-50)
Num_split = 0
list_ = []
#output for loop
for i, split in enumerate(audio_splits):
    #path for outputs
    output_file = "Internship_project/Audio_splits/split{0}.wav".format(i)
    #print("Exporting file", output_file)
    split.export(output_file, format="wav")
    Num_split = i
    list_.append(output_file)
print("split completed")
#load pre-trained model and processor and tokenizer
model = Wav2Vec2ForCTC.from_pretrained(model_checkpoint)
processor = Wav2Vec2Processor.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#list for file path output and output.txt for transcriptions
test_array = []
for i in range(0, Num_split):
    test_array.append("Internship_project/Audio_splits/split"+str(i)+".wav")
output = open("Internship_project/output.txt", "w")
# begin loop for passing hugging face a audio file from array then getting timestamps
for x in range(0, Num_split):
    #ensure audio is in 16000 for model
    speech, sample_rate = librosa.load(test_array[x],sr=16000)
    input_values = processor(speech, sampling_rate=sample_rate, return_tensors="pt").input_values
    logits = model(input_values).logits[0]
    pred_ids = torch.argmax(logits, axis=-1)
    # retrieve word stamps (analogous commands for `output_char_offsets`)
    outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
    # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = model.config.inputs_to_logits_ratio / sample_rate
    word_offsets = [
    {
        "word": d["word"],
        "start_time": (d["start_offset"] * time_offset),
        "end_time": (d["end_offset"] * time_offset),
    }
    for d in outputs.word_offsets
    ]
    print (list_[x], file = output)
    print (word_offsets, file = output)
print("done, output.txt populated")
