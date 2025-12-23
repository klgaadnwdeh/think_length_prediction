# 模型思维长度预测
中心思想:大模型在不联网，不增加外部知识的情况下，其面对问题的思维长度是可以被预测出来。

其他工作:预测大模型面对问题输出的步骤数目。
### 配置

实验设备:RTX-3090系列的GPU。

系统采用ubuntu-linux系统，请读者自行安装。
```
sudo apt update
sudo ubuntu-drivers autoinstall
sudo apt install build-essential
sudo apt install nvidia-cuda-toolkit
```

项目采用Anaconda，虚拟环境python配置版本要求>=3.7小于3.11，请读者在项目中运行以下命令完成环境配置。
```
conda create --name project_env python=3.10
pip install -r requirements.txt
```
另外需要读者安装[llamafactory库](https://github.com/hiyouga/LLaMA-Factory),[trl库](https://github.com/huggingface/trl)，实验所使用的trl库的版本为v0.25.1，读者可以使用该版本的库或者更新的版本并完成相应的配置。建议可以在Anaconda环境中创建两个虚拟环境完成llamafactory和trl的配置。

实验采用的模型如下所示，请自行下载并安装，安装源可选择[魔塔社区](https://www.modelscope.cn/home)。

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/f1b29471-be79-47a9-bf90-a1628c36568c" />

参数来源于阿里公布的官方文档,点击以下[链接](https://qwenlm.github.io/blog/qwen2/),进行阅读。

相关算法文档可在以下网址中下载:

GRPO:https://arxiv.org/abs/2402.03300

RLOO:https://arxiv.org/abs/2402.14740

KTO:https://arxiv.org/abs/2402.01306

DPO:https://arxiv.org/abs/2305.18290



### 思维token范围预测  

#### 1.运行下面的命令，命令会自动下载实验所需的数据集并进行数据处理，处理后保存数据到对应的子文件夹下。

```python
python ./think/get_think_process_1.py
python ./think/get_think_process_2.py
python ./think/get_think_process_3.py
```

#### 2.将处理好的数据，使用以下的python代码进行转换，转换为llamafactory支持的json格式。具体操作可查看项目中`./think/convert_json.py`中的代码进行配置相应的参数。

```python
python ./think/convert_json.py --choice 0 --data_path xxxx
```

#### 3.请在llamafactory库中训练对应的模型，模型的训练参数，参见

```
./data/llamafactory/think/think_parameter.txt
```

#### 4.训练好模型后，接下来需要修改trl库中的对应算法示例。因为实验将trl库中的相关算法示例文件进行了修改，请读者按照下面的文件对应trl库中的算法示例进行修改。

请参考
```bash
./trl-fix/GRPO.py
./trl-fix/GRPO.py
./trl-fix/kto.py
./trl-fix/DPO.py
```
对于GRPO算法,RLOO算法,KTO算法,DPO算法的示例代码文件进行修改。

#### 注明:
##### 2.ORPO算法和DPO算法需要将数据转换为偏好数据集的格式,数据格式点击下方[链接](https://hugging-face.cn/docs/trl/dataset_formats)进行查看。用以下的代码来实现数据转换。

```bash
./think/data_convert_preference.py
```

训练过程中，如有需要，请开启记录模式，记录实验过程。本实验使用tensorboard来记录实验过程。具体设置参考[链接](https://hugging-face.cn/docs/trl/logging)来实现,

#### 5训练过后，测试代码如下，具体操作可查看项目中`/think/think_predict.py`中的代码进行配置相应的参数

```bash
python /think/think_predict.py 
```

请参考论文实验结果，对照自己的实验结果,相差在较小误差范围内是完全正确的。

### 2模型输出的步骤数目预测

#### 1.首先下载并且运行如下示例
```bash
python ./step/create_step.py
python convert_json.py
```

#### 2.运用llamafactory进行训练模型，模型参数在

```bash
./data/llamafactory/step/step_paramter.txt
```

#### 3.训练完毕之后,运行

```bash
python /step/think_predict.py
```

