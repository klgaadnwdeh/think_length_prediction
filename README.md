# think_length_prediction
训练模型预测大模型输出的token区域和步骤数目
配置环境
思维token范围预测
1.本次实验，需要首先微调模型，请准备好对应的qwen2系列的0.5B,1.5B,7B模型。
2.下载对应的数据集合，处理过后，进行保存到对应的子文件夹下
```bash
python ./think/get_think_process_1.py
python ./think/get_think_process_2.py
python ./think/get_think_process_3.py
```

3.本次实验需要llamafactory库，请读者自行下载[huggingface库](https://github.com/hiyouga/LLaMA-Factory),
4将下载好的数据，使用以下的python代码进行转换为llamafactory支持的json格式,如以下命令
```bash
python ./think/convert_json.py --choice 0 
```
具体操作可查看对应的./think/convert_json.py中的具体操作。
5请在llamafactory库中训练对应的模型，模型的训练参数，参见
```bash
./data/llamafactory/think/think_parameter.txt
```
6训练好的模型后，请读者自行下载[trl库](https://github.com/huggingface/trl)
7下载完成后，需要对于trl库中的算法示例代码进行修改
请参考
```bash
./trl-fix/GRPO.py
./trl-fix/GRPO.py
./trl-fix/kto.py
./trl-fix/DPO.py
```
对于GRPO算法,RLOO算法,Kto算法,DPO算法的示例代码进行修改
注明:
1
GRPO和RLOO算法中，由于我们将探索式奖励转换为监督式奖励，因此需要引入这个模型预测一次的正确标签，因此需要第7步中的GRPO算法和RLOO算法代码进行调试，在训练过程中，进行如下操作
```bash

```
```bash

```
2kto算法和DPO算法需要将数据转换为偏好数据集的格式,数据格式参考https://hugging-face.cn/docs/trl/dataset_formats，用以下的代码来实现数据转换
```bash
data_convert_preference.py
```
训练过程中，请如有需要，请开始记录模式，具体参考https://hugging-face.cn/docs/trl/logging,实验这里使用tensorboard。
8训练过后，测试代码如下
```bash
python /think/think_predict.py
```
请参考论文实验结果，对照自己的实验结果，相差应该不大
2模型输出的步骤数目预测
首先下载并且运行如下示例
```bash
python ./step/create_step.py
python convert_json.py
```
3运用llamafactory进行训练模型，模型参数在
```bash
./data/llamafactory/step/step_paramter.txt
```
4训练完毕之后,运行
```bash
python /step/think_predict.py
```
