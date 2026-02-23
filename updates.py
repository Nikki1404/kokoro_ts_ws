us-central1-docker.pkg.dev/emr-dgt-autonomous-uctr1-snbx/cx-speech
now i need to push the docker image in this path via cloud run 
docker build us-central1-docker.pkg.dev/emr-dgt-autonomous-uctr1-snbx/cx-speech/kokor_ws:1.0.0 .
docker build us-central1-docker.pkg.dev/emr-dgt-autonomous-uctr1-snbx/cx-speech/kokor_openai:1.0.0 .

gcloud config set project emr-dgt-autonomous-uctr1-snbx
gcloud auth configure-docker us-central1-docker.pkg.dev
docker tag kokoro_openai_tts:latest \
us-central1-docker.pkg.dev/emr-dgt-autonomous-uctr1-snbx/cx-speech/kokoro_openai:1.0.0

docker tag cx_speech_tts_kokoro:latest \
us-central1-docker.pkg.dev/emr-dgt-autonomous-uctr1-snbx/cx-speech/kokoro_ws:1.0.0
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/emr-dgt-autonomous-uctr1-snbx/cx-speech/kokoro_openai:1.0.0
