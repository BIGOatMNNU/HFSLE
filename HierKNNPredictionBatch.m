%% Date: 2018-2-8
% �޸�˵����
% 1. ����������ݼ�ֻѡ10%20%30%40%50%������
% 2. ��������SVM����c=1����
% 3. ͼ������ݼ�ѡ20%���������������ݼ�ѡ10%����
function [accuracyMean, accuracyStd, F_LCAMean, FHMean, TIEmean, TestTime] = HierKNNPredictionBatch(data_array, tree, feature,i)
[m, numFeature] = size(data_array);
numFeature = numFeature - 1; 
numFolds  = 10;
k = 1;
baseline = 0;
% Test all features (baseline)
% Test 50% 40%	30%	20%	10% features.
% numsel_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4];
if strcmp(i,'DD') |strcmp(i,'F194')
    numSeleted = 40;
else
    numSeleted = round(numFeature * 0.2);
end

nrlen = length(numSeleted);
for j = 1:nrlen
    
    accuracyMean(1, k) = numSeleted;
    accuracyStd(1, k) = numSeleted;
    F_LCAMean(1, k) = numSeleted;
    FHMean(1, k) = numSeleted;
    TIEmean(1, k) = numSeleted;
    TestTime(1, k) = numSeleted;
    rand('seed', 1);
    indices = crossvalind('Kfold', m, numFolds);
    tic;
    [accuracyMean(2, k), accuracyStd(2, k), F_LCAMean(2, k), FHMean(2, k), TIEmean(2, k)] = FS_Kflod_TopDownKNNClassifier(data_array, numFolds, tree, feature, numSeleted, indices);
  
    TestTime(2, k) = toc;
    k = k+1;
   
end
%  if (baseline == 1)
%     accuracyMean(1, k) = numFeature;
%     accuracyStd(1, k) = numFeature;
%     F_LCAMean(1, k) = numFeature;
%     FHMean(1, k) = numFeature;
%     TIEmean(1, k) = numFeature;
%     TestTime(1, k) = numFeature;
%     rand('seed', 1);
%     indices = crossvalind('Kfold', m, numFolds);
%     tic;
%     [accuracyMean(2, k), accuracyStd(2, k), F_LCAMean(2, k), FHMean(2, k), TIEmean(2, k)] = FS_Kflod_TopDownSVMClassifier(data_array, numFolds, DAG, feature, numFeature, indices);
%     TestTime(2, k) = toc;
%  end
end
