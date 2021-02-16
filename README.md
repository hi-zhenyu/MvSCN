## Multi-view Spectral Clustering Network

Simple implementation of our paper MvSCN.
The details can be found in the IJCAI2019 paper [here](https://www.ijcai.org/proceedings/2019/356).

### parpare

- Python 3.6
- pip install -r ./requirements.txt
- maybe you need to make some change to keras, change "Python36\lib\site-packages\keras\engine\topology.py(line 3339)" from "original_keras_version = f.attrs['keras_version'].decode('utf8')" to "original_keras_version = f.attrs['keras_version']", change "Python36\lib\site-packages\keras\engine\topology.py(line 3343)" from "original_backend = f.attrs['backend'].decode('utf8')" to "original_backend = f.attrs['backend']"

### run

```
python ./run.py
```

### Citation

If you find our approach useful in your research, please consider citing:

```

@inproceedings{huang2019mvscn,
	Author = {Huang, Zhenyu and Zhou, Joey Tianyi and Peng, Xi and Zhang, Changqing and Zhu, Hongyuan and Lv, Jiancheng},
	Booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
	publisher = {International Joint Conferences on Artificial Intelligence Organization},
	Month = {10--16 Aug},
	Pages = {2563--2569},
	Title = {Multi-view Spectral Clustering Network},
	Year = {2019},
}
```
