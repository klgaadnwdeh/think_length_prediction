# 训练模型预测大模型输出的token区域和步骤数目 

配置环境

系统采用ubuntu-linux系统，请读者自行安装。

项目采用的python版本>=3.10，请读者在项目中运行以下命令完成环境配置。
```
pip install -r requirements.txt
```
另外需要读者安装[llamafactory库](https://github.com/hiyouga/LLaMA-Factory),[trl库](https://github.com/huggingface/trl)，并完成相应的配置。

实验采用的模型如下所示，请自行下载并安装，安装源可选择[魔塔社区](https://www.modelscope.cn/home)

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/f1b29471-be79-47a9-bf90-a1628c36568c" />

参数来源于阿里公布的官方文档,点击以下[链接](https://qwenlm.github.io/blog/qwen2/),进行阅读。



## 思维token范围预测  

### 1.本次实验，模型配置为



### 2.运行下面的命令，命令会自动下载实验所需的数据集并进行数据处理，处理后保存到对应的子文件夹下。

```bash
python ./think/get_think_process_1.py
python ./think/get_think_process_2.py
python ./think/get_think_process_3.py
```

### 3.本次实验需要llamafactory库，请读者自行下载，并且完成相应的配置。

### 4将处理好的数据，使用以下的python代码进行转换，转换为llamafactory支持的json格式。具体操作可查看对应的`./think/convert_json.py`中的代码。

```bash
python ./think/convert_json.py --choice 0 
```

### 5请在llamafactory库中训练对应的模型，模型的训练参数，参见

```bash
./data/llamafactory/think/think_parameter.txt
```

### 6训练好模型后，请读者自行下载，并进行配置。

### 7实验需要对于trl库中的算法代码进行修改

请参考
```bash
./trl-fix/GRPO.py
./trl-fix/GRPO.py
./trl-fix/kto.py
./trl-fix/DPO.py
```
对于GRPO算法,RLOO算法,KTO算法,DPO算法的示例代码进行修改

#### 注明:

##### 1GRPO和RLOO算法中，由于我们将探索式奖励转换为监督式奖励，因此需要引入问题对应的正确标签，并在奖励函数中将`reward_funcs`赋值为我们设置的reward函数。

```bash
GRPO中
def reward(prompts,completions, labels, **kwargs):
    """
    Args:e
        label: Tensor 或其他形式的真实标签
        completions: list[Tensor]，模型生成的文本列表，每个元素是一个位于 GPU 上的张量

    Returns:
        Tensor: 每个 completion 对应的奖励值
    """
    def clean_data(data):
        cleaned = []
        for item in data:
            try:
                # 尝试将元素转换为浮点数
                num = float(item)
                cleaned.append(num)
            except (ValueError, TypeError):
                if isinstance(item, str):  # 确保 item 是字符串类型
                    # 使用正则表达式查找所有连续的数字
                    # numbers = re.findall(r'\d+', item)diyige
                    numbers = re.findall(r'-?\d+\.?\d*', item)
                    if numbers:
                        # 假设我们只对第一个找到的数字感兴趣，并将其转换为浮点数
                        num = float(numbers[0])
                        cleaned.append(num)
                    else:
                        # 如果没有找到任何数字，则添加 NaN
                        cleaned.append(float('nan'))
                else:
                    # 如果既不是数值类型也不是字符串类型，则添加 NaN
                    cleaned.append(float('nan'))
        return cleaned

    # 清理并处理预测值
    # 首先将每个张量从 GPU 移动到 CPU 并转换为 NumPy 数组

    preds_list = [completion[0]["content"] for completion in
                  completions]
    # 如果 preds_list 中有非数值类型的数据，则需要进一步处理
    preds_cleaned = clean_data(preds_list)
    preds = np.nan_to_num(preds_cleaned, nan=0.0)
    preds = np.array(preds, dtype=np.float64)
    # 处理 labels，确保其与 preds 长度一致
    label = [label for label in
             labels]
    label = clean_data(label)
    labels = np.array(label, dtype=np.float64)
    # 计算奖励
    rewards = [
        12 if r == a else
        6 if abs(r - a) <= 1 else
        3 if abs(r - a) <= 2 else
        0.0
        for r, a in zip(labels, preds)
    ]
    #第一个数据集
    # rewards = [
    #         5 if abs(r - a) <=0 else
    #         4 if abs(r - a) <=0.5 else
    #         3 if abs(r - a) <= 1 else
    #         2 if abs(r - a) <= 1.5 else
    #         1 if abs(r - a) <= 2 else
    #         0.0
    #         for r, a in zip(labels, preds)]#第三个数据集
    # 第三个数据集
    # rewards = [
    #     10 if abs(r - a) <= 0 else
    #     8 if abs(r - a) <= 0.5 else
    #     6 if abs(r - a) <= 1 else
    #     4 if abs(r - a) <= 1.5 else
    #     2 if abs(r - a) <= 2 else
    #     0.0
    #     for r, a in zip(labels, preds)]第二个数据集
    print(rewards)#打印出模型一个批次获得的奖励得分
    return rewards
```
```bash
RLOO

def reward(labels, completions):
    """
    Args:
        labels: Tensor 或其他形式的真实标签
        completions: list[Tensor]，模型生成的文本列表，每个元素是一个位于 GPU 上的张量

    Returns:
        Tensor: 每个 completion 对应的奖励值
    """

    def clean_data(data):
        cleaned = []
        for item in data:
            try:
                num = float(item)
                cleaned.append(num)
            except (ValueError, TypeError):
                if isinstance(item, str):
                    numbers = re.findall(r'\d+', item)
                    if numbers:
                        num = float(numbers[0])
                        cleaned.append(num)
                    else:
                        cleaned.append(float('nan'))
                else:
                    cleaned.append(float('nan'))
        return cleaned

    preds_list = [completion.cpu().numpy() if isinstance(completion, torch.Tensor) else completion for completion in
                  completions]
    preds_cleaned = clean_data(preds_list)
    preds = np.nan_to_num(preds_cleaned, nan=0.0)
    preds = np.array(preds, dtype=np.float64)
    label = [label.cpu().numpy() if isinstance(label, torch.Tensor) else label for label in labels]
    label = clean_data(label)
    labels = np.array(label, dtype=np.float64)
    rewards = [
        12 if r == a else
        6 if abs(r - a) <= 1 else
        3 if abs(r - a) <= 2 else
        0.0
        for r, a in zip(labels, preds)
    ]
    #第一个数据集
    # rewards = [
    #     5 if abs(r - a) <= 0 else
    #     4 if abs(r - a) <= 0.5 else
    #     3 if abs(r - a) <= 1 else
    #     2 if abs(r - a) <= 1.5 else
    #     1 if abs(r - a) <= 2 else
    #     0.0
    #     for r, a in zip(labels, preds)]  # 第三个数据集
    # rewards = [
    #     16 if abs(r - a) < 0 else
    #     12 if abs(r - a) < 0.5 else
    #     8 if abs(r - a) <= 1 else
    #     4 if abs(r - a) <= 2 else
    #     0.0
    #     for r, a in zip(labels, preds)
    # ]第二个数据集
    print(rewards)#打印出模型一个批次获得的奖励得分
    return torch.tensor(rewards, dtype=torch.float)
```

##### 2kto算法和DPO算法需要将数据转换为偏好数据集的格式,[数据格式](https://hugging-face.cn/docs/trl/dataset_formats),，用以下的代码来实现数据转换。

```bash
data_convert_preference.py
```

训练过程中，请如有需要，请开始记录模式，[参考](https://hugging-face.cn/docs/trl/logging),实验这里使用tensorboard。

### 8训练过后，测试代码如下

```bash
python /think/think_predict.py
```

请参考论文实验结果，对照自己的实验结果，相差应该不大。

## 2模型输出的步骤数目预测

### 1.首先下载并且运行如下示例
```bash
python ./step/create_step.py
python convert_json.py
```

### 2.运用llamafactory进行训练模型，模型参数在

```bash
./data/llamafactory/step/step_paramter.txt
```

### 3.训练完毕之后,运行

```bash
python /step/think_predict.py
```
