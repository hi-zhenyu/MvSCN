## Multi-view Spectral Clustering Network

Simple implementation of our paper MvSCN.
The details can be found in the IJCAI2019 paper [here](https://www.ijcai.org/proceedings/2019/356).

Note: this work was foked from user `hi-zhenyu/MvSCN`, and tested for academic purposes by

CHAALAL Mohamed 

TALANTIKITE Salma 

KESRAOUI Elias

### requirements
- tensorflow==1.14.0
- keras==2.0.8
- PyYAML==5.1.1
- protobuf==3.20
- scikit-learn==0.21.2
- munkres==1.1.4
- pytz==2019.1
- h5py==2.10.0

## How to run
## Option 1: using the repository locally  
## 1. Clone the repo
```
git clone -b requirements https://github.com/elmondhir/MvSCN.git
cd MvSCN
```

## 2. Create a virtual enviroments

### Using ven

```
python3.7 -m venv "my_env_name"
my_env_name\Scripts\activate.bat
```
### Using conda
```
conda create -n "my_env_name" python=3.7
conda activate my_env_name
```

## 3. install the requierment

```
pip install -r requirements.txt
```

## 4. Launch run.py

```
python run.py
```

## Option 2: using docker
Make sure Docker is installed on your machine. If not, you can download and install Docker from the [official Docker website](https://www.docker.com/get-started).


2.1. Build the Docker image:

    ```
    docker-compose build
    ```

2.2. Start the application:

    ```
    docker-compose up
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
