# Kaggle Dsbowl
https://www.kaggle.com/c/data-science-bowl-2018
https://www.kaggle.com/takuok/keras-generator-starter-lb-0-326(公開したkernel)

# Usage
```
python3 main.py
```

モデルの切り替えはmain.pyで読み込むモデルを変える。
コードは以下の部分
```
from model.unet import Unet

model = Unet(img_size)
```

# Installation
初回起動前に以下のコマンドを実行すること
```
pip install requirements.txt
```
