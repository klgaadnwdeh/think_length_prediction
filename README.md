
# CoT长度感知的大语言模型推理调度机制

核心思想：在不联网、不引入外部知识的前提下，大语言模型（LLM）面对特定问题时所“思考”的 token 长度（即内部推理步数）是可以被有效预测的。

本项目不仅探索了思维长度的预测，还进一步研究了模型输出步骤数目的预测，为理解 LLM 的内部决策过程提供新视角。

## 📚 相关工作

思维长度建模：预测模型在生成答案前所需的内部推理 token 数量。
步骤数目预测：估计模型完成任务所需的逻辑/操作步骤总数。
本项目复现并改进了多种主流对齐算法（GRPO、RLOO、KTO、DPO 等），相关论文见下文。

## ⚙️ 环境配置指南

硬件与系统要求
GPU：NVIDIA RTX 3090 系列（或同等算力）
操作系统：Ubuntu Linux（建议 20.04 或 22.04）
安装基础依赖（Ubuntu）

```sudo apt update
sudo ubuntu-drivers autoinstall          # 自动安装显卡驱动
sudo apt install build-essential         # 编译工具链
sudo apt install nvidia-cuda-toolkit     # CUDA 工具包
```
💡 提示：若已安装 NVIDIA 官方驱动，可跳过 ubuntu-drivers 步骤。

Python 环境搭建（推荐使用 Anaconda）创建虚拟环境（Python 版本需满足 3.7 ≤ version < 3.11）：

```
conda create --name think_predict python=3.10 -y
conda activate think_predict
```

安装项目依赖：
```
pip install -r requirements.txt
```
LLaMA-Factory(使用最新版本即可)

TRL (Transformer Reinforcement Learning)

注意：实验基于 trl v0.25.1，建议使用该版本或更新版本。为避免依赖冲突，推荐为 LLaMA-Factory 和 TRL 分别创建独立虚拟环境。

模型准备
实验所用基础模型来自 Qwen 系列，请通过以下任一方式获取：

🌐 魔搭社区（ModelScope）：https://www.modelscope.cn/home
📄 官方技术博客：Qwen2 技术详解（含参数说明）

📖 相关算法文献
算法	论文链接
相关算法文档可在以下网址中下载:

GRPO:https://arxiv.org/abs/2402.03300

RLOO:https://arxiv.org/abs/2402.14740

ORPO:https://arxiv.org/pdf/2403.07691

DPO:https://arxiv.org/abs/2305.18290

## 🔍 实验一：思维 Token 范围预测
### 步骤 1：数据预处理

自动下载并处理原始数据集，结果将保存至子目录：


```
python ./think/get_think_process_1.py
python ./think/get_think_process_2.py
python ./think/get_think_process_3.py
```
### 步骤 2：格式转换（适配 LLaMA-Factory）

将处理后的数据转为标准 JSON 格式：


```python ./think/convert_json.py --choice 0 ```

📌 参数说明详见 （./think/convert_json.py）。

### 步骤 3：模型训练（使用 LLaMA-Factory）

训练配置文件位于：（可以查看./llamafactory/think/parameter.txt或者./data/llamafactory/think/think_parameter.txt） 按照该配置启动训练即可。

### 步骤 4：修改 TRL 算法示例（关键！）
由于本项目对 TRL 的算法实现进行了定制化修改，请替换以下文件到你的 TRL 库对应位置：

```
./trl-fix/GRPO.py      → 替换 TRL 中的 GRPO 示例
./trl-fix/RLOO.py      → 替换 RLOO 示例
./trl-fix/ORPO_unsloth.py → 替换 ORPO 示例
./trl-fix/DPO_unsloth.py  → 替换 DPO 示例
```
⚠️ 注意：ORPO 与 DPO 需要偏好数据格式。格式规范见 TRL 数据格式文档，转换脚本为：
```
python ./think/data_convert_preference.py
```
步骤 5：实验记录（推荐开启）
使用 TensorBoard 记录训练过程,在训练配置中添加：
```
report_to = "tensorboard"
logging_dir = "./logs"
```
详细配置参考：TRL 日志文档

### 步骤 6：推理预测
训练完成后，运行预测脚本：
```
python /think/think_predict.py
```
📌 参数配置详见脚本内注释。

✅ 预期结果应与论文报告值在合理误差范围内一致。

## 🔢 实验二：模型输出步骤数目预测
### 步骤 1：生成步骤数据
```
python ./step/create_step.py
python ./step/convert_json.py   # 注意：此处应为 step 目录下的脚本
```
### 步骤 2：训练模型
使用 LLaMA-Factory，配置文件位于： (./data/llamafactory/step/step_parameter.txt)

### 步骤 3：执行预测
```
python /step/think_predict.py
```
### 🙌 致谢与引用
如果你觉得本项目对你有帮助，欢迎 Star ⭐ 并引用相关论文。

如有疑问或改进建议，欢迎提交 Issue 或 PR！

温馨提示：所有路径请根据你的实际项目结构调整；敏感信息（如 Token、密钥）切勿提交至代码库
