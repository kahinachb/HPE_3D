# HPE_3D
use python 3.10
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
add .torchscript from nlf repo (nlf model)
export yolo models : yolo export model=yolov11m.pt format=engine device=0

