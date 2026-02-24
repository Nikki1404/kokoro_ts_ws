for this 

from openai import OpenAI

client = OpenAI(
    base_url="https://kokoro-openai-tts-150916788856.us-central1.run.app/v1",
    api_key="not-needed",
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_sky",
    input="Hello world from Kokoro and this is generating from kokoro!",
    response_format="mp3",
) as resp:
    resp.stream_to_file("output4.mp3")

print("Saved -> output4.mp3")

I can only hear
hello world from and this is generating from 
why kokoro is not being tts

input="Hey Wait—did you hear that? I thought I heard a knock. It was a sunny day; however, we decided to stay inside."
