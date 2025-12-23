# CoT长度感知的大语言模型推理调度机制
中心思想:
（1）大模型在不联网，不增加外部知识的情况下，其面对问题的思维Token长度是可以被预测出来。
（2）将模型的思维链式输出转为逻辑化步骤数目进行预测，可以极大的提高预测准确率。
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
# CoT长度感知的大语言模型推理调度机制

核心思想：在不联网、不引入外部知识的前提下，大语言模型（LLM）面对特定问题时所“思考”的 token 长度（即内部推理步数）是可以被有效预测的。

本项目不仅探索了思维长度的预测，还进一步研究了模型输出步骤数目的预测，为理解 LLM 的内部决策过程提供新视角。

##📚 相关工作
思维长度建模：预测模型在生成答案前所需的内部推理 token 数量。
步骤数目预测：估计模型完成任务所需的逻辑/操作步骤总数。
本项目复现并改进了多种主流对齐算法（GRPO、RLOO、KTO、DPO 等），相关论文见下文。
##⚙️ 环境配置指南
硬件与系统要求
GPU：NVIDIA RTX 3090 系列（或同等算力）
操作系统：Ubuntu Linux（建议 20.04 或 22.04）
安装基础依赖（Ubuntu）
bash
编辑
```sudo apt update
sudo ubuntu-drivers autoinstall          # 自动安装显卡驱动
sudo apt install build-essential         # 编译工具链
sudo apt install nvidia-cuda-toolkit     # CUDA 工具包
```
💡 提示：若已安装 NVIDIA 官方驱动，可跳过 ubuntu-drivers 步骤。

Python 环境搭建（推荐使用 Anaconda）创建虚拟环境（Python 版本需满足 3.7 ≤ version < 3.11）：
bash
编辑
```conda create --name think_predict python=3.10 -y
conda activate think_predict```
安装项目依赖：
bash
编辑pip install -r requirements.txt
安装关键库：
LLaMA-Factory
TRL (Transformer Reinforcement Learning)
注意：实验基于 trl v0.25.1，建议使用该版本或更新版本。为避免依赖冲突，推荐为 LLaMA-Factory 和 TRL 分别创建独立虚拟环境。

模型准备
实验所用基础模型来自 Qwen 系列，请通过以下任一方式获取：

🌐 魔搭社区（ModelScope）：https://www.modelscope.cn/home
📄 官方技术博客：Qwen2 技术详解（含参数说明）

📖 相关算法文献
算法	论文链接
```
GRPO	arXiv:2402.03300
RLOO	arXiv:2402.14740
KTO	arXiv:2402.01306
DPO	arXiv:2305.18290```
###🔍 实验一：思维 Token 范围预测
####步骤 1：数据预处理
自动下载并处理原始数据集，结果将保存至子目录：

bash
编辑
```python ./think/get_think_process_1.py
python ./think/get_think_process_2.py
python ./think/get_think_process_3.py```
####步骤 2：格式转换（适配 LLaMA-Factory）
将处理后的数据转为标准 JSON 格式：

bash
编辑
```python ./think/convert_json.py --choice 0 --data_path /your/data/path```
📌 参数说明详见 ./think/convert_json.py。

步骤 3：模型训练（使用 LLaMA-Factory）
训练配置文件位于：
./data/llamafactory/think/think_parameter.txt
按照该配置启动训练即可。
步骤 4：修改 TRL 算法示例（关键！）
由于本项目对 TRL 的算法实现进行了定制化修改，请替换以下文件到你的 TRL 库对应位置：

text
编辑
./trl-fix/GRPO.py      → 替换 TRL 中的 GRPO 示例
./trl-fix/RLOO.py      → 替换 RLOO 示例
./trl-fix/kto.py       → 替换 KTO 示例
./trl-fix/DPO.py       → 替换 DPO 示例
⚠️ 注意：ORPO 与 DPO 需要偏好数据格式。格式规范见 TRL 数据格式文档，转换脚本为：

bash
编辑
python ./think/data_convert_preference.py
步骤 5：实验记录（推荐开启）
使用 TensorBoard 记录训练过程：

python
编辑
# 在训练配置中添加：
report_to = "tensorboard"
logging_dir = "./logs"
详细配置参考：TRL 日志文档

步骤 6：推理预测
训练完成后，运行预测脚本：

bash
编辑
python /think/think_predict.py
📌 参数配置详见脚本内注释。

✅ 预期结果应与论文报告值在合理误差范围内一致。

🔢 实验二：模型输出步骤数目预测
步骤 1：生成步骤数据
bash
编辑
python ./step/create_step.py
python ./step/convert_json.py   # 注意：此处应为 step 目录下的脚本
步骤 2：训练模型
使用 LLaMA-Factory，配置文件位于：
./data/llamafactory/step/step_parameter.txt
步骤 3：执行预测
bash
编辑
python /step/think_predict.py
🙌 致谢与引用
如果你觉得本项目对你有帮助，欢迎 Star ⭐ 并引用相关论文。

如有疑问或改进建议，欢迎提交 Issue 或 PR！

温馨提示：所有路径请根据你的实际项目结构调整；敏感信息（如 Token、密钥）切勿提交至代码库
