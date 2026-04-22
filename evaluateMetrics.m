function metrics = evaluateMetrics(yTrue, yPred, setName)
%EVALUATEMETRICS 计算回归预测的常用评估指标
%   yTrue   : 真实值向量
%   yPred   : 预测值向量
%   setName : 数据集名称（用于打印）
%
%   返回结构体 metrics，包含：
%     RMSE  — 均方根误差
%     MAE   — 平均绝对误差
%     MAPE  — 平均绝对百分比误差（仅统计 yTrue > 1 的点）
%     R2    — 决定系数 R²
%     maxAE — 最大绝对误差

errors = yTrue - yPred;

metrics.RMSE  = sqrt(mean(errors.^2));
metrics.MAE   = mean(abs(errors));
metrics.maxAE = max(abs(errors));

% MAPE 仅在真实值较大时有意义（避免零值附近膨胀）
validIdx = abs(yTrue) > 1;
if any(validIdx)
    metrics.MAPE = mean(abs(errors(validIdx) ./ yTrue(validIdx))) * 100;
else
    metrics.MAPE = NaN;
end

SSres = sum(errors.^2);
SStot = sum((yTrue - mean(yTrue)).^2);
metrics.R2 = 1 - SSres / SStot;

fprintf('  [%s]  RMSE=%.4f | MAE=%.4f | MAPE=%.2f%% | R²=%.4f | MaxAE=%.4f\n', ...
    setName, metrics.RMSE, metrics.MAE, metrics.MAPE, metrics.R2, metrics.maxAE);

end
