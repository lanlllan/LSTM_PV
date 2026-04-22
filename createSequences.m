function [X, Y, seqIdx] = createSequences(features, target, seqLen, predLen)
%CREATESEQUENCES 将时间序列数据转换为 LSTM 所需的滑动窗口格式
%   features : N × numFeatures 归一化特征矩阵
%   target   : N × 1 归一化目标向量
%   seqLen   : 输入窗口长度
%   predLen  : 预测步长偏移
%
%   返回值:
%   X      : cell 数组，每个元素为 numFeatures × seqLen 矩阵
%   Y      : cell 数组，每个元素为标量（预测目标值）
%   seqIdx : 每条样本对应的原始目标时间点索引

N = size(features, 1);
numSamples = N - seqLen - predLen + 1;

X      = cell(numSamples, 1);
Y      = cell(numSamples, 1);
seqIdx = zeros(numSamples, 1);

for i = 1:numSamples
    X{i}      = features(i : i+seqLen-1, :)';   % numFeatures × seqLen
    Y{i}      = target(i + seqLen + predLen - 1); % 标量
    seqIdx(i) = i + seqLen + predLen - 1;
end

end
