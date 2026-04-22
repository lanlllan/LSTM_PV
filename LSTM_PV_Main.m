%% LSTM 光伏出力预测模型 —— 主程序
%  数据来源：光伏数据.xlsx（15 min 分辨率，2021-01 ~ 2023-06）
%  输入特征：压强、湿度、云量、温度、风速、辐照量
%  输出目标：有功功率（MW）
%  环境要求：MATLAB R2021b+ 及 Deep Learning Toolbox

clc; clear; close all;
rng(42);  % 固定随机种子，保证可复现

%% ======================== 1. 超参数配置 ========================
cfg.seqLen       = 16;       % 输入时间步长（16 × 15min = 4h 历史窗口）
cfg.predLen      = 1;        % 预测步长（提前 1 步 = 15min）
cfg.trainRatio   = 0.7;      % 训练集比例
cfg.valRatio     = 0.15;     % 验证集比例（剩余为测试集）
cfg.lstmUnits    = [128 64]; % 两层 LSTM 隐藏单元数
cfg.dropout      = 0.2;      % Dropout 比例
cfg.maxEpochs    = 100;      % 最大训练轮数
cfg.miniBatch    = 256;      % 小批量大小
cfg.initLR       = 0.001;    % 初始学习率
cfg.lrDropPeriod = 30;       % 学习率衰减周期
cfg.lrDropFactor = 0.5;      % 学习率衰减因子
cfg.patience     = 15;       % 早停耐心值

featureNames = {'压强','湿度','云量','温度','风速','辐照量'};
targetName   = '有功功率';

%% ======================== 2. 数据读取 ========================
fprintf('>>> 正在读取数据...\n');
dataFile = '光伏数据.xlsx';
if ~isfile(dataFile)
    error('未找到数据文件：%s，请确认文件在当前目录下。', dataFile);
end

T = readtable(dataFile, 'VariableNamingRule', 'preserve');
fprintf('    共读取 %d 条记录，时间范围：%s ~ %s\n', ...
    height(T), string(T.date(1)), string(T.date(end)));

timeStamps = T.date;
featureData = T{:, featureNames};  % N × 6
targetData  = T{:, {targetName}};  % N × 1
numFeatures = size(featureData, 2);

%% ======================== 3. 数据预处理 ========================
fprintf('>>> 正在预处理数据...\n');

% 3.1 缺失值处理（线性插值）
featureData = fillmissing(featureData, 'linear');
targetData  = fillmissing(targetData,  'linear');

% 3.2 Min-Max 归一化到 [0, 1]
[featureNorm, featureParams] = mapminmax(featureData', 0, 1);
featureNorm = featureNorm';  % 转回 N × numFeatures

[targetNorm, targetParams] = mapminmax(targetData', 0, 1);
targetNorm = targetNorm';    % 转回 N × 1

% 3.3 构建滑动窗口序列
[X, Y, seqIdx] = createSequences(featureNorm, targetNorm, cfg.seqLen, cfg.predLen);
fprintf('    生成序列样本数：%d（seqLen=%d, predLen=%d）\n', ...
    length(Y), cfg.seqLen, cfg.predLen);

%% ======================== 4. 数据集划分 ========================
numSamples = length(Y);
numTrain = floor(numSamples * cfg.trainRatio);
numVal   = floor(numSamples * cfg.valRatio);
numTest  = numSamples - numTrain - numVal;

XTrain = X(1:numTrain);
YTrain = Y(1:numTrain);

XVal = X(numTrain+1 : numTrain+numVal);
YVal = Y(numTrain+1 : numTrain+numVal);

XTest = X(numTrain+numVal+1 : end);
YTest = Y(numTrain+numVal+1 : end);

testTimeIdx = seqIdx(numTrain+numVal+1 : end);

fprintf('    训练集：%d | 验证集：%d | 测试集：%d\n', numTrain, numVal, numTest);

%% ======================== 5. 构建 LSTM 网络 ========================
fprintf('>>> 正在构建 LSTM 网络...\n');

layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input', ...
        'Normalization', 'none')

    lstmLayer(cfg.lstmUnits(1), 'OutputMode', 'sequence', ...
        'Name', 'lstm1')
    dropoutLayer(cfg.dropout, 'Name', 'drop1')

    lstmLayer(cfg.lstmUnits(2), 'OutputMode', 'last', ...
        'Name', 'lstm2')
    dropoutLayer(cfg.dropout, 'Name', 'drop2')

    fullyConnectedLayer(32, 'Name', 'fc1')
    reluLayer('Name', 'relu1')

    fullyConnectedLayer(1, 'Name', 'fc_out')
    regressionLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs',           cfg.maxEpochs, ...
    'MiniBatchSize',       cfg.miniBatch, ...
    'InitialLearnRate',    cfg.initLR, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropPeriod', cfg.lrDropPeriod, ...
    'LearnRateDropFactor', cfg.lrDropFactor, ...
    'ValidationData',      {XVal, cell2mat(YVal)}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience',  cfg.patience, ...
    'Shuffle',             'every-epoch', ...
    'GradientThreshold',   1, ...
    'Verbose',             true, ...
    'VerboseFrequency',    20, ...
    'Plots',               'training-progress', ...
    'ExecutionEnvironment','auto');

%% ======================== 6. 模型训练 ========================
fprintf('>>> 开始训练 LSTM 模型...\n');
tic;
net = trainNetwork(XTrain, cell2mat(YTrain), layers, options);
trainTime = toc;
fprintf('>>> 训练完成！用时 %.1f 秒\n', trainTime);

%% ======================== 7. 模型预测 ========================
fprintf('>>> 正在预测...\n');

YPredNorm_train = predict(net, XTrain, 'MiniBatchSize', cfg.miniBatch);
YPredNorm_val   = predict(net, XVal,   'MiniBatchSize', cfg.miniBatch);
YPredNorm_test  = predict(net, XTest,  'MiniBatchSize', cfg.miniBatch);

% 反归一化
YPred_train = mapminmax('reverse', YPredNorm_train', targetParams)';
YTrue_train = mapminmax('reverse', cell2mat(YTrain)', targetParams)';

YPred_val = mapminmax('reverse', YPredNorm_val', targetParams)';
YTrue_val = mapminmax('reverse', cell2mat(YVal)', targetParams)';

YPred_test = mapminmax('reverse', YPredNorm_test', targetParams)';
YTrue_test = mapminmax('reverse', cell2mat(YTest)', targetParams)';

% 功率非负约束
YPred_train = max(YPred_train, 0);
YPred_val   = max(YPred_val, 0);
YPred_test  = max(YPred_test, 0);

%% ======================== 8. 评估指标 ========================
fprintf('\n========== 模型评估结果 ==========\n');

[metrics_train] = evaluateMetrics(YTrue_train, YPred_train, '训练集');
[metrics_val]   = evaluateMetrics(YTrue_val,   YPred_val,   '验证集');
[metrics_test]  = evaluateMetrics(YTrue_test,  YPred_test,  '测试集');

%% ======================== 9. 可视化 ========================
fprintf('>>> 正在生成可视化图表...\n');
testTime = timeStamps(testTimeIdx);

plotPredictionResults(testTime, YTrue_test, YPred_test, ...
    metrics_test, cfg, featureNames);

%% ======================== 10. 保存模型 ========================
save('LSTM_PV_Model.mat', 'net', 'featureParams', 'targetParams', ...
    'cfg', 'featureNames', 'targetName', 'metrics_test');
fprintf('>>> 模型已保存至 LSTM_PV_Model.mat\n');

fprintf('\n>>> 全部流程完成！\n');
