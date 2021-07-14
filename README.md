# Jetson_nano安装Tensorflow（2.3.1）、Pytorch（1.8.0）、paddlepaddle（2.0.0）
这是用来从系统安装到部署Tensorflow（2.3.1）、Pytorch（1.8.0）、paddlepaddle（2.1.0）的文章

## 1.系统安装及配置部分
[Jetson Nano开发者套件入门](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)
下载 Jetson Nano 开发者套件 SD 卡镜像，以及相关烧录软件，进行镜像烧录。推荐32G卡以上（系统一开始占掉14G）

烧录完成后进入系统完成Ubuntu的首次开机设置

#### 增加交换空间大小
由于显存内存共享，即使本身设置有2G的交换空间，建议还是多增加2G。

 1. 增加2G交换空间
 	`sudo fallocate -l 2G /swapfile`
 	
 2. 所有用户都可以读写swap file,设置正确的权限
 	`sudo chmod 666 /swapfile`
 	
 3. 设置交换空间
	`sudo mkswap /swapfile`
	
 4. 激活交换空间
	`sudo swapon /swapfile`
	为了使这个激活永久有效
	`sudo vi /etc/fstab`
	打开文件在最末尾粘贴
	`/swapfile swap swap defaults 0 0`
	
 5. 验证增加空间是否有效

```bash
sudo swapon --show
sudo free -h
```



#### 删除不必要的软件以节省空间
```bash
sudo apt-get purge libreoffice*
sudo apt-get clean
```
	
更新一下软件源：
`sudo apt-get update`
	
如果更新太慢或者更新失败的话就需要更换安装源（我这里不需要更换）
	
更新软件：
`sudo apt-get update`

#### 添加nvcc -V 到环境
`sudo vim ~/.bashrc`

在文件末尾添加三行代码：
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda
```
保存退出再输入命令：
`sudo source ~/.bashrc`

验证一下，查看CUDA版本：
`nvcc -V`

## 2.为安装深度框架的准备软件
#### 安装重要的软件包
```bash
sudo apt-get install build-essential make cmake cmake-curses-gui
sudo apt-get install git g++ pkg-config curl  zip zlib1g-dev libopenblas-base libopenmpi-dev 
sudo apt-get install libatlas-base-dev gfortran libcanberra-gtk-module libcanberra-gtk3-module
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev
sudo apt-get install nano locate screen

sudo apt-get install libfreetype6-dev 
sudo apt-get install protobuf-compiler libprotobuf-dev openssl
sudo apt-get install libssl-dev libcurl4-openssl-dev
sudo apt-get install cython3

sudo apt-get install libxml2-dev libxslt1-dev

sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libxvidcore-dev libavresample-dev
sudo apt-get install libtiff-dev libjpeg-dev libpng-dev

#sudo apt-get install -y python3-dev python3-testresources python3-setuptools python3-pip

python3 -m pip install --upgrade pip
```

#### 默认python命令指向python3
```bash
update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1

update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
```

#### 系统工具jtop和nvidia-smi的结合
可以同时查看Jetson Nano的CPU和GPU资源以及温度
```bash
sudo -H pip install jetson-stats
sudo jtop
```

## 3.安装Tensorflow-gpu 2.3.1版本
#### 安装机器学习常用包
```bash
sudo apt install python3-scipy -y
sudo apt install python3-pandas -y
sudo apt install python3-sklearn -y
sudo apt install python3-seaborn -y
```
#### 检测CUDA是否正常安装
`nvcc -V`

#### 安装HDF5
`sudo apt install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran`

#### 安装python依赖包
`sudo pip3 install -U numpy==1.18.5 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11`

#### 安装tensorflow-gpu
`sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow`

#### 测试tensorflow
```python
import tensorflow as tf
# 输出提示：
# 2020-10-11 15:25:36.253267: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.2

a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)
# 输出结果：
# tf.Tensor(3.0, shape=(), dtype=float32)
 
print('GPU:', tf.test.is_gpu_available())
# 输出最后一句为：
# GPU: True
```
#### 安装Keras
`sudo pip3 install keras`

完成

## 4.安装Pytorch 1.8.0
需要下载[nvidia提供的pytorch安装文件](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-7-0-now-available/72048)
已上传到网盘自行下载:
链接：https://pan.baidu.com/s/1B42hAHZVduoZmc1A1h3AkA 
提取码：bi46
```bash
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
pip3 install numpy
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
测试是否安装成功
```bash
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
```
#### 安装 torchvision

说明地址：[https://github.com/pytorch/vision](https://github.com/pytorch/vision)
`sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev`
因为我安装的是1.8.0的pytorch，对应 torchvision 版本是0.9.0

手动去这个地址下载相关版本代码包，解压后进入文件夹
```bash
export BUILD_VERSION=0.9.0
sudo python3 setup.py install 
```
测试是否安装成功
`import torchvision`
`print(torchvision.__version__)`

## 5.安装paddlepaddle 2.0.0
下载[编译好的paddlepaddle](https://paddle-inference-dist.cdn.bcebos.com/temp_data/paddlepaddle_gpu-0.0.0-cp36-cp36m-linux_aarch64.whl)

然后pip安装Jetson Nano版本的Paddle Inference。（也可进行源码编译安装）
[https://blog.csdn.net/weixin_45449540/article/details/107704028](https://blog.csdn.net/weixin_45449540/article/details/107704028)

`pip install paddlehub`

