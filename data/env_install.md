## 以公共机房电脑WSL为例（南雍楼西550）

### miniconda安装

```bash
# 进入用户目录
cd /home/[username]

# 在用户目录下创建miniconda3文件夹
mkdir -p ~/miniconda3
# 下载miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# 安装miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# 删除安装包
rm -rf ~/miniconda3/miniconda.sh
```

手动在`~/.bashrc` 中添加如下指令：

```bash
# 在文件末尾添加以下内容
# 注意需要将[username]换成提供的用户名！！！

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/[usrname]/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/[usrname]/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/[usrname]/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/[usrname]/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

### 配置conda镜像源（`vim ~/.condarc`）：

 ```bash
 channels:
   - defaults
 show_channel_urls: true
 default_channels:
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
 custom_channels:
   conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
 ```

###   配置虚拟环境：

```bash
conda create -n proj2 python=3.12
conda activate proj2
pip install torch==2.8.0 torchvision==0.23.0 triton==3.4.0 \
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

测试以下代码输出"OK"即正常：

```bash
python -c "import torch; x = torch.randn(1000,1000).cuda(); y = x @ x; torch.cuda.synchronize(); print('OK', y.shape, torch.cuda.get_device_name(0))"
```

### 安装C编译器

```bash
conda install gcc_linux-64 gxx_linux-64

# 装好后配置环境变量：
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

确认以下命令输出正常：

```bash
which $CC
$CC --version
```



### 其余配置（optional）：

#### 网络代理：

##### option A: 

校园网内有一台电脑已配置好VPN，并且设置中“允许局域网”，可以设置代理如下（也可写入~/.bashrc）

```bash
export PROXY=<your ip address>

export HTTP_PROXY=$PROXY
export HTTPS_PROXY=$PROXY
export ALL_PROXY=$PROXY
export http_proxy=$PROXY
export https_proxy=$PROXY
export all_proxy=$PROXY
```

##### option B: 

安装VPN软件并导入自己的订阅，建议使用`TUN/虚拟网卡`模式。



#### codex：

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

#### ClaudeCode：

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

