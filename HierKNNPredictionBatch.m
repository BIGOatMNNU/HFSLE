function [accuracyMean, accuracyStd, F_LCAMean, FHMean, TIEmean, TestTime] = HierKNNPredictionBatch(data_array, tree, feature,i)
[m, numFeature] = size(data_array);
numFeature = numFeature - 1; 
numFolds  = 10;
k = 1;
baseline = 0;

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
end
