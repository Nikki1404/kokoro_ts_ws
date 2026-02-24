after deplying to cloud run 
got this url https://kokoro-openai-tts-150916788856.us-central1.run.app
client.py 
from openai import OpenAI

client = OpenAI(
    base_url="https://kokoro-openai-tts-150916788856.us-central1.run.app",
    api_key="not-needed",
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",                  # or "tts-1" (ignored by server)
    voice="af_sky+af_bella",         # or "af_heart"
    input="Hello world from Kokoro!",
    response_format="mp3",
) as resp:
    resp.stream_to_file("output3.mp3")

print("Saved -> output3.mp3")

and when testing from local got this error 
(kokoro_env) PS C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app> python .\test_openai.py
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\fastapi_impl\app\test_openai.py", line 8, in <module>
    with client.audio.speech.with_streaming_response.create(
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model="kokoro",                  # or "tts-1" (ignored by server)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        response_format="mp3",
        ^^^^^^^^^^^^^^^^^^^^^^
    ) as resp:
    ^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_response.py", line 626, in __enter__
    self.__response = self._request_func()
                      ~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\resources\audio\speech.py", line 104, in create
    return self._post(
           ~~~~~~~~~~^
        "/audio/speech",
        ^^^^^^^^^^^^^^^^
    ...<15 lines>...
        cast_to=_legacy_response.HttpxBinaryResponseContent,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1297, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Downloads\cx-speech-tts-main\cx-speech-tts-main\kokoro\basic_impl\client\kokoro_env\Lib\site-packages\openai\_base_client.py", line 1070, in request
    raise self._make_status_error_from_response(err.response) from None
openai.NotFoundError: Error code: 404 - {'detail': 'Not Found'}


from openai import OpenAI

client = OpenAI(
    base_url="https://kokoro-openai-tts-150916788856.us-central1.run.app/v1",
    api_key="not-needed",
)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_sky+af_bella",
    input="Hello world from Kokoro!",
    response_format="mp3",
) as resp:
    resp.stream_to_file("output3.mp3")

print("Saved -> output3.mp3")
