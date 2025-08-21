download the index_demo folder from https://drive.google.com/drive/folders/1Y4gnvplLDlb5-wxB2W3M4QbQFi8mhrWL?usp=sharing

conda env create -f another_env.yml

then activate the environment by conda activate biz-agent-gpu-2

then run python -m spacy download en_core_web_sm

for those with nvidia gpus that support cuda, run

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118


