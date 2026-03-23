import whisper
import json

model = whisper.load_model("large-v2")
result = model.transcribe(audio = "audios/12_Exercise 1 - Pure HTML Media Player .mp3",
                          language="hi",
                          task="translate",
                          word_timestamps=False)

print(result['segments'])
chunks = []
for segment in result['segments']:
    chunk = {
        'start': segment['start'],
        'end': segment['end'],
        'text': segment['text'].strip()
    }
    chunks.append(chunk)

print(chunks)

with open('output.json', 'w') as f:
    json.dump(chunks, f)