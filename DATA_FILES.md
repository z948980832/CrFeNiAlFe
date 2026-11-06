# `ML` 目录结构及文件概要

工作目录：`D:\File\zzz\chaoyu\A2-multi\ML`

本目录按执行顺序编号为 `0_raw_data`、`1_SPXY`、`2_train`、`3_test`、`4_predict`，覆盖数据准备、固定划分、特征工程、模型训练、测试集评估和待预测数据推理。

## 1. 目录结构

```text
ML/
├─ DATA_FILES.md
├─ WORKFLOW.md
├─ 0_raw_data/
│  ├─ CrFeNi.csv
│  ├─ element.csv
│  ├─ Hij.csv
│  ├─ data.ipynb
│  └─ parameter.csv
├─ 1_SPXY/
│  ├─ spxy_split.py
│  └─ split_indices.json
├─ 2_train/
│  ├─ 2_1_feature_engineering/
│  │  ├─ feature_engineering.py
│  │  ├─ make_feature_heatmaps.py
│  │  ├─ engineered_features_86.csv
│  │  ├─ selected_features_55.csv
│  │  ├─ feature_target_correlations.csv
│  │  └─ feature_target_heatmap.png
│  └─ 2_2_model_training/
│     ├─ fcnn_model.py
│     ├─ train_fcnn.py
│     ├─ export_training_history.py
│     ├─ requirements.txt
│     ├─ fcnn_model.pt
│     ├─ metrics.json
│     ├─ training_history.csv
│     └─ training_loss_curve.png
├─ 3_test/
│  ├─ test_fcnn.py
│  ├─ export_spxy_test_results.py
│  ├─ plot_test_predictions.py
│  ├─ test_predictions.csv
│  ├─ spxy_test_results_full.csv
│  └─ test_predictions_scatter.png
└─ 4_predict/
   ├─ predict_fcnn.py
   ├─ plot_pre_vs_exp.py
   ├─ pre_with_predictions.csv
   ├─ pre_vs_exp.csv
   └─ pre_vs_exp_scatter.png
```

脚本运行后可能产生 `__pycache__/` 和 `.pyc` 文件。它们是 Python 自动生成的字节码缓存，不属于源代码、模型参数或正式结果。

## 2. 目录职责

| 目录                           | 职责                                                                         |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `0_raw_data/` | 保存原始成分与性能数据、元素物性、二元混合焓、描述符计算笔记本和最终训练数据 |
| `1_SPXY/` | 生成并保存训练集、验证集和测试集的固定划分 |
| `2_train/2_1_feature_engineering/` | 构造候选特征、筛选模型输入特征、导出特征矩阵和相关性图 |
| `2_train/2_2_model_training/` | 定义并训练全连接神经网络，保存模型参数、指标和训练历史 |
| `3_test/` | 使用保存的模型对 SPXY 留出测试集推理，导出逐行结果和测试散点图 |
| `4_predict/` | 对待预测样本推理，并保存预测结果及预测值—实验值对比图 |

## 3. 根目录文档

| 文件              | 概要                                           |
| ----------------- | ---------------------------------------------- |
| `DATA_FILES.md` | 当前目录结构、文件用途、主要结果和执行入口     |
| `WORKFLOW.md`   | 从数据准备到模型预测的完整方法、参数和结果说明 |

## 4. `0_raw_data`：数据准备

| 文件                   |                规模 | 概要                                                                     |
| ---------------------- | ------------------: | ------------------------------------------------------------------------ |
| `0_raw_data/CrFeNi.csv` | 248行 × 12列 | 10种元素成分以及 UTS、EL，是训练样本的基础成分与性能数据 |
| `0_raw_data/element.csv` | 7项物性 × 10种元素 | 原子半径、熔点、密度、杨氏模量、电负性、VEC 和体积模量等元素属性 |
| `0_raw_data/Hij.csv` | 44条记录 | 元素对及其二元混合焓，用于计算合金混合焓相关描述符；文件没有表头 |
| `0_raw_data/data.ipynb` | 24个代码单元 | 根据元素成分、元素物性和二元混合焓计算物理化学描述符的原始数据处理笔记本 |
| `0_raw_data/parameter.csv` | 248行 × 20列 | 最终训练总数据集：A:R 为18个输入特征，最后两列为 UTS、EL |

`parameter.csv` 的18个输入包括10个元素成分和8个物理化学描述符：平均原子半径、原子尺寸差、电负性、电负性标准差、平均 VEC、VEC 标准差、混合熵和混合焓。

`data.ipynb` 中仍保留部分历史相对路径。若需要重新执行该笔记本，应先将其中的输入和输出路径调整为当前 `ML/0_raw_data` 目录。

## 5. `1_SPXY`：数据划分

| 文件                        | 类型 | 概要                                                    |
| --------------------------- | ---- | ------------------------------------------------------- |
| `1_SPXY/spxy_split.py` | 脚本 | 根据 A:R 输入和 UTS/EL 输出执行 SPXY 划分 |
| `1_SPXY/split_indices.json` | 结果 | 保存零基数据行索引：训练集190条、验证集21条、测试集37条 |

`split_indices.json` 是训练、测试和训练集相关性分析共用的固定划分依据。测试脚本只读取其中的 `test_indices`，不会重新划分数据。

## 6. `2_train/2_1_feature_engineering`：特征工程

| 文件                                | 类型 | 概要                                                                                                  |
| ----------------------------------- | ---- | ----------------------------------------------------------------------------------------------------- |
| `feature_engineering.py`          | 脚本 | 保留 A:R 的18个原始特征，并构造元素存在性、组成统计量和二阶成分交互项；同时提供训练集筛选和标准化函数 |
| `make_feature_heatmaps.py`        | 脚本 | 导出86维和55维特征矩阵、计算 Pearson 相关系数并绘制圆形相关性图；`--plot-only` 可只更新图片         |
| `engineered_features_86.csv`      | 数据 | 248条样本的86个候选特征：18个原始特征、10个存在性特征、3个组成统计特征和55个二阶交互候选项            |
| `selected_features_55.csv`        | 数据 | 248条样本的55个最终模型输入特征，即31个固定特征加24个训练集筛选出的交互项                             |
| `feature_target_correlations.csv` | 结果 | 55个模型输入特征分别与 UTS、EL 的 Pearson 相关系数                                                    |
| `feature_target_heatmap.png` | 图片 | 只绘制 A:R 的18个原始特征；红色表示正相关，蓝色表示负相关，半径表示 `|r|`，对应位置标注 Pearson `r` |

两个特征矩阵均保持与 `0_raw_data/parameter.csv` 相同的248条样本顺序。相关性 CSV 保留55个模型输入特征，但圆形图只显示18个原始特征。

## 7. `2_train/2_2_model_training`：模型训练

| 文件                           | 类型 | 概要                                                                                  |
| ------------------------------ | ---- | ------------------------------------------------------------------------------------- |
| `fcnn_model.py`              | 脚本 | 定义共享主干、残差全连接块和 UTS/EL 双输出头                                          |
| `train_fcnn.py`              | 脚本 | 读取数据、调用 SPXY 划分和特征工程、执行6次固定初始化训练、验证集选模并保存模型与指标 |
| `export_training_history.py` | 脚本 | 使用最终初始化种子重现训练过程并导出逐轮 loss 数据和曲线                              |
| `requirements.txt`           | 配置 | Python 依赖：`numpy<2`、PyTorch 和 Matplotlib                                       |
| `fcnn_model.pt`              | 模型 | 网络权重、18个原始输入名称、55维特征索引与名称、输入/输出标准化参数和训练元数据       |
| `metrics.json`               | 结果 | 数据规模、划分信息、6个初始化候选、最终模型和训练/验证/测试指标                       |
| `training_history.csv`       | 结果 | 最终模型对应的702轮训练 loss、验证 loss、学习率和最佳值记录                           |
| `training_loss_curve.png`    | 图片 | 训练集与验证集 loss 曲线，并标出最佳轮次202                                           |

当前模型输入维数为55，输出固定为 UTS 和 EL。

## 8. `3_test`：留出测试集推理

| 文件                             | 类型 | 概要                                                                                  |
| -------------------------------- | ---- | ------------------------------------------------------------------------------------- |
| `test_fcnn.py`                 | 脚本 | 加载`fcnn_model.pt`，对 `split_indices.json` 中的37条测试记录推理，不重新训练模型 |
| `export_spxy_test_results.py` | 脚本 | 将简版预测结果与 `0_raw_data/parameter.csv` 中的原始 A:R 特征合并 |
| `plot_test_predictions.py`     | 脚本 | 根据测试结果绘制 UTS、EL 的预测值—真实值散点图                                       |
| `test_predictions.csv`         | 结果 | 37条测试记录、7列，包含真实值、预测值和有符号误差                                     |
| `spxy_test_results_full.csv`   | 结果 | 37条测试记录、28列，包含数据索引、A:R、真实值、预测值、误差和绝对误差                 |
| `test_predictions_scatter.png` | 图片 | 左侧为 UTS、右侧为 EL；横轴为真实值，纵轴为预测值，虚线为`y=x`                      |

当前测试集指标：

| 输出 |      R² |    RMSE |     MAE |
| ---- | -------: | ------: | ------: |
| UTS  | 0.936489 | 76.4060 | 58.5950 |
| EL   | 0.867389 |  7.1357 |  4.0992 |

## 9. `4_predict`：待预测数据推理

| 文件                         | 类型 | 概要                                                                       |
| ---------------------------- | ---- | -------------------------------------------------------------------------- |
| `predict_fcnn.py`          | 脚本 | 读取具有相同 A:R 表头的 CSV，加载保存模型并推理 UTS 和 EL                  |
| `pre_with_predictions.csv` | 结果 | 12条待预测记录；保留18个原始特征，并将`pre_UTS`、`pre_EL` 作为最后两列 |
| `pre_vs_exp.csv`           | 数据 | 12条样本的旧模型预测、新模型预测和实验值，用于外部结果对比                 |
| `plot_pre_vs_exp.py`       | 脚本 | 读取`pre_vs_exp.csv` 最后四列并绘制新模型预测值—实验值散点图            |
| `pre_vs_exp_scatter.png`   | 图片 | 12条样本的 UTS、EL 预测值—实验值散点图                                    |

`3_test/` 和 `4_predict/` 使用同一个模型参数，但用途不同：`3_test/` 评估固定的 SPXY 留出测试集，`4_predict/` 面向独立待预测数据。
