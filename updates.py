docker build \
  --build-arg http_proxy=http://163.116.128.80:8080 \
  --build-arg https_proxy=http://163.116.128.80:8080 \
  -t kokoro_openai_tts .

docker build \
  --build-arg http_proxy=http://163.116.128.80:8080 \
  --build-arg https_proxy=http://163.116.128.80:8080 \
  -t kokoro_ws .
