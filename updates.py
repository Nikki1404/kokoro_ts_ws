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
Get "https://us-central1-docker.pkg.dev/v2/": read tcp 10.90.126.61:42656->172.253.124.82:443: read: connection reset by peer
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# gcloud builds submit \
  --tag us-central1-docker.pkg.dev/emr-dgt-autonomous-uctr1-snbx/cx-speech/kokoro_openai:1.0.0
ERROR: (gcloud.builds.submit) Invalid value for [source]: Dockerfile required when specifying --tag
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav# cd cx-speech-tts/fastapi_impl_gpu/
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/cx-speech-tts/fastapi_impl_gpu# gcloud builds submit   --tag us-central1-docker.pkg.dev/emr-dgt-autonomous-uctr1-snbx/cx-speech/kokoro_openai:1.0.0
Creating temporary archive of 16 file(s) totalling 19.0 KiB before compression.
Uploading tarball of [.] to [gs://emr-dgt-autonomous-uctr1-snbx_cloudbuild/source/1771862524.677656-93e817ef77574e948aae5100c83a93b5.tgz]
Created [https://cloudbuild.googleapis.com/v1/projects/emr-dgt-autonomous-uctr1-snbx/locations/global/builds/e6e5fc64-16f9-4789-bca8-a850aceb858f].
Logs are available at [ https://console.cloud.google.com/cloud-build/builds/e6e5fc64-16f9-4789-bca8-a850aceb858f?project=150916788856 ].
Waiting for build to complete. Polling interval: 1 second(s).
-------------------------------------------------------- REMOTE BUILD OUTPUT --------------------------------------------------------
starting build "e6e5fc64-16f9-4789-bca8-a850aceb858f"

FETCHSOURCE
Fetching storage object: gs://emr-dgt-autonomous-uctr1-snbx_cloudbuild/source/1771862524.677656-93e817ef77574e948aae5100c83a93b5.tgz#1771862557442538
Copying gs://emr-dgt-autonomous-uctr1-snbx_cloudbuild/source/1771862524.677656-93e817ef77574e948aae5100c83a93b5.tgz#1771862557442538...
/ [1 files][  7.7 KiB/  7.7 KiB]
Operation completed over 1 objects/7.7 KiB.
BUILD
Already have image (with digest): gcr.io/cloud-builders/docker
Sending build context to Docker daemon   34.3kB
Step 1/19 : FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
12.1.1-runtime-ubuntu22.04: Pulling from nvidia/cuda
aece8493d397: Pulling fs layer
dd4939a04761: Pulling fs layer
b0d7cc89b769: Pulling fs layer
1532d9024b9c: Pulling fs layer
04fc8a31fa53: Pulling fs layer
a14a8a8a6ebc: Pulling fs layer
7d61afc7a3ac: Pulling fs layer
8bd2762ffdd9: Pulling fs layer
2a5ee6fadd42: Pulling fs layer
7d61afc7a3ac: Waiting
8bd2762ffdd9: Waiting
2a5ee6fadd42: Waiting
1532d9024b9c: Verifying Checksum
1532d9024b9c: Download complete
04fc8a31fa53: Verifying Checksum
04fc8a31fa53: Download complete
dd4939a04761: Verifying Checksum
dd4939a04761: Download complete
aece8493d397: Verifying Checksum
aece8493d397: Download complete
7d61afc7a3ac: Verifying Checksum
7d61afc7a3ac: Download complete
2a5ee6fadd42: Verifying Checksum
2a5ee6fadd42: Download complete
8bd2762ffdd9: Verifying Checksum
8bd2762ffdd9: Download complete
b0d7cc89b769: Verifying Checksum
b0d7cc89b769: Download complete
aece8493d397: Pull complete
dd4939a04761: Pull complete
b0d7cc89b769: Pull complete
1532d9024b9c: Pull complete
04fc8a31fa53: Pull complete
a14a8a8a6ebc: Verifying Checksum
a14a8a8a6ebc: Download complete
a14a8a8a6ebc: Pull complete
7d61afc7a3ac: Pull complete
8bd2762ffdd9: Pull complete
2a5ee6fadd42: Pull complete
Digest: sha256:8bbc6e304b193e84327fa30d93eea70ec0213b808239a46602a919a479a73b12
Status: Downloaded newer image for nvidia/cuda:12.1.1-runtime-ubuntu22.04
 ---> 0495908f9381
Step 2/19 : ENV https_proxy="http://163.116.128.80:8080"
 ---> Running in a07cccc61a19
Removing intermediate container a07cccc61a19
 ---> 248e8f58fc2c
Step 3/19 : ENV http_proxy="http://163.116.128.80:8080"
 ---> Running in 10ea4e0cfbd2
Removing intermediate container 10ea4e0cfbd2
 ---> 48ab123f0bb8
Step 4/19 : ENV DEBIAN_FRONTEND=noninteractive
 ---> Running in f171ae06cf62
Removing intermediate container f171ae06cf62
 ---> c7fea4fe14ec
Step 5/19 : WORKDIR /app
 ---> Running in 955dce3da3ac
Removing intermediate container 955dce3da3ac
 ---> 9dec3bb493a1
Step 6/19 : RUN apt-get update && apt-get install -y     git     ffmpeg     python3     python3-pip     python3-dev     build-essential     && rm -rf /var/lib/apt/lists/*
 ---> Running in ef691ef1cc40
Ign:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Ign:2 http://archive.ubuntu.com/ubuntu jammy InRelease
Ign:3 http://security.ubuntu.com/ubuntu jammy-security InRelease
Ign:4 http://archive.ubuntu.com/ubuntu jammy-updates InRelease
Ign:5 http://archive.ubuntu.com/ubuntu jammy-backports InRelease
Ign:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Ign:2 http://archive.ubuntu.com/ubuntu jammy InRelease
Ign:3 http://security.ubuntu.com/ubuntu jammy-security InRelease
Ign:4 http://archive.ubuntu.com/ubuntu jammy-updates InRelease
Ign:5 http://archive.ubuntu.com/ubuntu jammy-backports InRelease
Ign:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Ign:2 http://archive.ubuntu.com/ubuntu jammy InRelease
Ign:3 http://security.ubuntu.com/ubuntu jammy-security InRelease
Ign:4 http://archive.ubuntu.com/ubuntu jammy-updates InRelease
Ign:5 http://archive.ubuntu.com/ubuntu jammy-backports InRelease
Err:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
  Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
Err:2 http://archive.ubuntu.com/ubuntu jammy InRelease
  Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
Err:3 http://security.ubuntu.com/ubuntu jammy-security InRelease
  Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
Err:4 http://archive.ubuntu.com/ubuntu jammy-updates InRelease
  Unable to connect to 163.116.128.80:8080:
Err:5 http://archive.ubuntu.com/ubuntu jammy-backports InRelease
  Unable to connect to 163.116.128.80:8080:
Reading package lists...
W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/jammy/InRelease  Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/jammy-updates/InRelease  Unable to connect to 163.116.128.80:8080:
W: Failed to fetch http://archive.ubuntu.com/ubuntu/dists/jammy-backports/InRelease  Unable to connect to 163.116.128.80:8080:
W: Failed to fetch http://security.ubuntu.com/ubuntu/dists/jammy-security/InRelease  Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
W: Failed to fetch https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/InRelease  Could not connect to 163.116.128.80:8080 (163.116.128.80), connection timed out
W: Some index files failed to download. They have been ignored, or old ones used instead.
Reading package lists...
Building dependency tree...
Reading state information...
E: Unable to locate package git
E: Unable to locate package ffmpeg
E: Unable to locate package python3
E: Unable to locate package python3-pip
E: Unable to locate package python3-dev
E: Unable to locate package build-essential
The command '/bin/sh -c apt-get update && apt-get install -y     git     ffmpeg     python3     python3-pip     python3-dev     build-essential     && rm -rf /var/lib/apt/lists/*' returned a non-zero code: 100
ERROR
ERROR: build step 0 "gcr.io/cloud-builders/docker" failed: step exited with non-zero status: 100

-------------------------------------------------------------------------------------------------------------------------------------

BUILD FAILURE: Build step failure: build step 0 "gcr.io/cloud-builders/docker" failed: step exited with non-zero status: 100
ERROR: (gcloud.builds.submit) build e6e5fc64-16f9-4789-bca8-a850aceb858f completed with status "FAILURE"
