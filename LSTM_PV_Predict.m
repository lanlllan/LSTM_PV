%% LSTM 光伏出力预测 —— 预测脚本
%
% =========================================================================
%  脚本名称：LSTM_PV_Predict.m
%  功    能：加载已训练的 LSTM 模型，对新的光伏气象数据进行有功功率预测
%  配套训练：LSTM_PV_Main.m（需先运行以生成模型文件）
% =========================================================================
%
%  【前置条件】
%    1. 已运行 LSTM_PV_Main.m，在当前目录下生成 LSTM_PV_Model.mat
%       该文件包含：训练好的网络 net、归一化参数 featureParams / targetParams、
%       超参数配置 cfg、特征列名 featureNames、目标列名 targetName
%    2. MATLAB R2021b 或更高版本，且已安装 Deep Learning Toolbox
%
%  【输入文件要求】
%    格式：.xlsx（Excel 表格）
%    必需列（7 列）：
%      - date       : 时间戳（datetime 类型，15 min 间隔）
%      - 压强       : 大气压强
%      - 湿度       : 相对湿度
%      - 云量       : 云量
%      - 温度       : 气温
%      - 风速       : 风速
%      - 辐照量     : 太阳辐照量
%    可选列（1 列）：
%      - 有功功率   : 实际有功功率（MW）。若存在，脚本会自动计算评估指标
%                     （RMSE / MAE / MAPE / R² / MaxAE）并绘制对比图
%
%  【输出文件】
%    prediction_result.xlsx，包含以下列：
%      - 时间              : 每条预测对应的时间戳
%      - 预测有功功率_MW   : 模型预测值
%      - 真实有功功率_MW   : 真实值（仅当输入含"有功功率"列时）
%      - 误差_MW           : 真实值 − 预测值（仅当输入含"有功功率"列时）
%
%  【使用方法】
%    1. 将本脚本第 46 行的 inputFile 改为待预测的 Excel 文件路径
%    2. 直接运行本脚本即可
%
%  【处理流程】
%    读取 Excel → 校验列名 → 缺失值线性插值 → 使用训练参数归一化
%    → 滑动窗口切片（窗口长度 = cfg.seqLen = 16 步 = 4 小时）
%    → LSTM 前向推理 → 反归一化 → 非负截断 → 保存结果
%
%  【注意事项】
%    - 滑动窗口需要前 seqLen + predLen − 1 条数据作为历史上下文，
%      因此输出行数 = 输入行数 − (seqLen + predLen − 1)
%    - 预测使用的归一化参数来自训练集，确保新数据量纲与训练一致
%    - 功率输出强制非负（max(pred, 0)）
% =========================================================================

clc; clear; close all;

%% ======================== 1. 加载模型 ========================
modelFile = 'LSTM_PV_Model.mat'; % ← 模型文件路径（由 LSTM_PV_Main.m 生成）
if ~isfile(modelFile)
    error('未找到模型文件：%s，请先运行 LSTM_PV_Main.m 完成训练。', modelFile);
end

fprintf('>>> 正在加载模型...\n');
load(modelFile, 'net', 'featureParams', 'targetParams', 'cfg', 'featureNames', 'targetName');
fprintf('    模型加载成功（seqLen=%d, predLen=%d）\n', cfg.seqLen, cfg.predLen);

%% ======================== 2. 读取待预测数据 ========================
inputFile = '光伏数据.xlsx';   % ← 替换为待预测的 Excel 文件路径
if ~isfile(inputFile)
    error('未找到输入文件：%s', inputFile);
end

fprintf('>>> 正在读取待预测数据...\n');
T = readtable(inputFile, 'VariableNamingRule', 'preserve');
fprintf('    共读取 %d 条记录，时间范围：%s ~ %s\n', ...
    height(T), string(T.date(1)), string(T.date(end)));

missingCols = setdiff(featureNames, T.Properties.VariableNames);
if ~isempty(missingCols)
    error('输入数据缺少以下特征列：%s', strjoin(missingCols, ', '));
end

timeStamps  = T.date;
featureData = T{:, featureNames};
numFeatures = size(featureData, 2);

hasTarget = ismember(targetName, T.Properties.VariableNames);
if hasTarget
    targetData = T{:, {targetName}};
    targetData = fillmissing(targetData, 'linear');
    fprintf('    检测到 "%s" 列，将同时输出评估指标\n', targetName);
end

%% ======================== 3. 数据预处理 ========================
fprintf('>>> 正在预处理数据...\n');

featureData = fillmissing(featureData, 'linear');

% 使用训练时保存的归一化参数
featureNorm = mapminmax('apply', featureData', featureParams)';

%% ======================== 4. 构建滑动窗口 ========================
numRows    = size(featureNorm, 1);
numSamples = numRows - cfg.seqLen - cfg.predLen + 1;

if numSamples <= 0
    error('数据量不足：至少需要 %d 行（seqLen=%d + predLen=%d），当前仅 %d 行。', ...
        cfg.seqLen + cfg.predLen, cfg.seqLen, cfg.predLen, numRows);
end

X      = cell(numSamples, 1);
seqIdx = zeros(numSamples, 1);
for i = 1:numSamples
    X{i}      = featureNorm(i : i+cfg.seqLen-1, :)';  % numFeatures × seqLen
    seqIdx(i) = i + cfg.seqLen + cfg.predLen - 1;
end

fprintf('    生成预测样本数：%d\n', numSamples);

%% ======================== 5. 模型预测 ========================
fprintf('>>> 正在预测...\n');
YPredNorm = predict(net, X, 'MiniBatchSize', 256);

% 反归一化
YPred = mapminmax('reverse', YPredNorm', targetParams)';
YPred = max(YPred, 0);  % 功率非负

predTime = timeStamps(seqIdx);

%% ======================== 6. 评估（若有真实值） ========================
if hasTarget
    targetNorm = mapminmax('apply', targetData', targetParams)';
    YTrue = targetData(seqIdx);

    fprintf('\n========== 预测评估结果 ==========\n');
    metrics = evaluateMetrics(YTrue, YPred, '预测集');

    figure('Name', '预测对比', 'Position', [100 100 1200 400]);
    plot(predTime, YTrue, 'b-', 'LineWidth', 0.8); hold on;
    plot(predTime, YPred, 'r-', 'LineWidth', 0.8);
    legend('真实值', '预测值', 'Location', 'best');
    xlabel('时间'); ylabel('有功功率 (MW)');
    title(sprintf('LSTM 预测 vs 真实  |  RMSE=%.4f  R²=%.4f', metrics.RMSE, metrics.R2));
    grid on; hold off;
end

%% ======================== 7. 保存结果 ========================
outputFile = 'prediction_result.xlsx';

resultTable = table(predTime, YPred, 'VariableNames', {'时间', '预测有功功率_MW'});
if hasTarget
    resultTable.('真实有功功率_MW') = targetData(seqIdx);
    resultTable.('误差_MW') = targetData(seqIdx) - YPred;
end

writetable(resultTable, outputFile);
fprintf('>>> 预测结果已保存至 %s（共 %d 条）\n', outputFile, numSamples);
fprintf('>>> 预测完成！\n');
