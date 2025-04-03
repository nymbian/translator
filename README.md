# translator
基于 Transformer 的中英文翻译模型demo，数据集比较小，主要用于理解Transformer的机制。

推荐使用uv
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```
初始化环境
```shell
uv venv --python 3.10
```
使用环境
```shell
source .venv/bin/activate
```
安装依赖
```shell
uv pip install -r requirements.txt
```
执行
```shell
python main.py
```
翻译测试
```shell
python translate.py
```
 
