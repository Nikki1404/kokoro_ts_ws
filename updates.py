from openai import OpenAI

client = OpenAI(
    base_url="https://kokoro-openai-tts-150916788856.us-central1.run.app/v1",
    api_key="not-needed",
)

response = client.audio.speech.create(
    model="kokoro",
    voice="af_sky",
    input="this is kokoro and gcp test",
    response_format="mp3",
)

with open("output_fixed.mp3", "wb") as f:
    f.write(response.content)

print("Saved -> output_fixed.mp3")
