## Preprocessing
```
pip install -r requirements.txt
```
#### Dataset
```
BDD100K: (https://bdd-data.berkeley.edu)
Dark_Zurich: (https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/)
```
#### weights
weight download link：https://drive.google.com/file/d/1g7_sxeEvwWo2jzgTn6QQ4tSG55Z3npES/view?usp=drive_link

## Detect
```
python tools/test.py --weights weights/light_multi.pth
```
## Acknowledgments
我们主要参考如下工作，感谢他们精彩的工作。YOLOP(https://github.com/hustvl/YOLOP)
Multinet(https://github.com/MarvinTeichmann/MultiNet)
