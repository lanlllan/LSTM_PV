# LSTM 光伏出力预测模型

基于长短期记忆网络（LSTM）的光伏电站有功功率预测系统，利用历史气象数据与功率输出数据进行时序建模，实现光伏出力的短期预测。

## 环境要求

- **MATLAB** R2021b 或更高版本
- **Deep Learning Toolbox**（LSTM 网络训练与推理）
- GPU 加速（可选，推荐使用 NVIDIA GPU 以加速训练）

## 项目结构

```
mxc/
├── 光伏数据.xlsx              # 原始数据文件（15 min 分辨率）
├── LSTM_PV_Main.m             # 主程序（一键运行）
├── createSequences.m          # 滑动窗口序列构造函数
├── evaluateMetrics.m          # 评估指标计算函数
├── plotPredictionResults.m    # 可视化绘图函数
├── LSTM_PV_Model.mat          # 训练后自动保存的模型文件
└── README.md                  # 本文件
```

## 数据说明

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| date | datetime | — | 时间戳（15 min 间隔） |
| 压强 | float | hPa | 大气压强 |
| 湿度 | float | % | 相对湿度 |
| 云量 | float | % | 云覆盖量 |
| 温度 | float | °C | 环境温度 |
| 风速 | float | m/s | 风速 |
| 辐照量 | float | W/m² | 太阳辐照强度 |
| 有功功率 | float | MW | **预测目标** |
| 额定功率 | int | MW | 电站额定容量（100 MW） |

**数据规模**：70,176 条记录，时间跨度 2021-01-11 至 2023-06-27（约 2.5 年）。

**特征相关性**：

| 特征 | Pearson 相关系数 | 影响方向 |
|------|:---:|------|
| 辐照量 | +0.897 | 最强正相关，光伏出力的核心驱动因素 |
| 湿度 | -0.494 | 较强负相关，高湿度降低辐照透射率 |
| 温度 | +0.278 | 正相关，温度高通常伴随日照充足 |
| 压强 | +0.270 | 正相关 |
| 风速 | +0.203 | 正相关，散热有助于维持组件效率 |
| 云量 | -0.064 | 弱负相关 |

## 模型架构

```
输入层 (6特征 × 16时间步)
    ↓
LSTM 层 1 (128 隐藏单元, sequence 输出)
    ↓
Dropout 层 (rate = 0.2)
    ↓
LSTM 层 2 (64 隐藏单元, last 输出)
    ↓
Dropout 层 (rate = 0.2)
    ↓
全连接层 (32 神经元) + ReLU 激活
    ↓
全连接层 (1 神经元) → 功率预测输出
```

**关键设计**：
- **双层 LSTM**：第一层捕捉低层时序特征并保留全序列信息，第二层提取高层语义并输出最终隐状态
- **Dropout 正则化**：防止过拟合，提升泛化能力
- **梯度裁剪**：阈值设为 1，防止 LSTM 训练中的梯度爆炸
- **功率非负约束**：预测结果强制非负，符合物理意义

## 快速开始

### 1. 准备环境

确保 4 个 `.m` 文件与 `光伏数据.xlsx` 位于同一目录。

### 2. 运行模型

在 MATLAB 命令窗口中执行：

```matlab
LSTM_PV_Main
```

程序将自动完成以下流程：
1. 读取 Excel 数据
2. Min-Max 归一化预处理
3. 滑动窗口序列构建
4. 按 70%/15%/15% 划分训练/验证/测试集
5. 构建并训练 LSTM 网络
6. 预测与反归一化
7. 计算评估指标（RMSE、MAE、MAPE、R²）
8. 生成可视化图表
9. 保存模型至 `LSTM_PV_Model.mat`

### 3. 调整超参数

在 `LSTM_PV_Main.m` 顶部的 `cfg` 结构体中修改：

```matlab
cfg.seqLen       = 16;       % 输入窗口长度（16步 = 4h）
cfg.predLen      = 1;        % 预测步长（1步 = 15min）
cfg.trainRatio   = 0.7;      % 训练集比例
cfg.valRatio     = 0.15;     % 验证集比例
cfg.lstmUnits    = [128 64]; % LSTM 隐藏单元数
cfg.dropout      = 0.2;      % Dropout 比例
cfg.maxEpochs    = 100;      % 最大训练轮数
cfg.miniBatch    = 256;      % 小批量大小
cfg.initLR       = 0.001;    % 初始学习率
cfg.patience     = 15;       % 早停耐心值
```

**调参建议**：
- 增大 `seqLen`（如 32/48）可捕捉更长期的天气变化趋势，但增加计算量
- 减小 `miniBatch`（如 64/128）可能提升收敛精度，但训练速度变慢
- 若过拟合明显，可增大 `dropout`（如 0.3/0.4）

## 输出说明

### 控制台输出

```
[训练集]  RMSE=x.xxxx | MAE=x.xxxx | MAPE=xx.xx% | R²=x.xxxx | MaxAE=xx.xxxx
[验证集]  RMSE=x.xxxx | MAE=x.xxxx | MAPE=xx.xx% | R²=x.xxxx | MaxAE=xx.xxxx
[测试集]  RMSE=x.xxxx | MAE=x.xxxx | MAPE=xx.xx% | R²=x.xxxx | MaxAE=xx.xxxx
```

### 可视化图表

| 图窗 | 内容 |
|------|------|
| 图 1（6 子图） | 预测对比全局图、局部放大（3天）、散点图、误差分布直方图、绝对误差时序、日内平均出力曲线 |
| 图 2 | 6 个特征与有功功率的 Pearson 相关系数柱状图 |
| 图 3 | LSTM 网络结构示意图 |

### 保存的模型文件

`LSTM_PV_Model.mat` 包含：
- `net`：训练好的 LSTM 网络对象
- `featureParams` / `targetParams`：归一化参数（用于新数据预测时的缩放）
- `cfg`：训练超参数
- `metrics_test`：测试集评估结果

## 模型复用（预测新数据）

```matlab
% 加载已训练模型
load('LSTM_PV_Model.mat');

% 准备新数据：newData 为 N × 6 矩阵 [压强, 湿度, 云量, 温度, 风速, 辐照量]
newNorm = mapminmax('apply', newData', featureParams)';

% 构建序列
[XNew, ~, ~] = createSequences(newNorm, zeros(size(newNorm,1),1), cfg.seqLen, cfg.predLen);

% 预测
YPredNorm = predict(net, XNew, 'MiniBatchSize', 256);
YPred = mapminmax('reverse', YPredNorm', targetParams)';
YPred = max(YPred, 0);  % 非负约束
```

## 评估指标说明

| 指标 | 公式 | 说明 |
|------|------|------|
| RMSE | √(Σ(y-ŷ)²/n) | 均方根误差，对大误差敏感 |
| MAE | Σ\|y-ŷ\|/n | 平均绝对误差，直观反映平均偏差 |
| MAPE | Σ\|y-ŷ\|/y × 100% | 平均绝对百分比误差（仅统计功率 > 1MW 的时段） |
| R² | 1 - SS_res/SS_tot | 决定系数，越接近 1 拟合越好 |
| MaxAE | max(\|y-ŷ\|) | 最大绝对误差，反映极端预测偏差 |
