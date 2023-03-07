clc; clear;
str1={'DD';'F194'};
rangea=[10 10];
rangeb=[100 10];
rangec=[10 0.01];
m = length(str1); 
rng('default');
for i = 1:m
    filename = [str1{i} 'Train']
    load (filename);
    [X,Y,Z]=create_SubTable2(data_array, tree);
    tic;
    [feature{i},W{i},YD{i}] = HFSLE(X, Y,Z, tree, 10, rangea(i), rangeb(i) ,rangec(i),1);
    TrainTime{i} =toc;
    %Test feature batch
    testFile = [str1{i}, 'Test.mat']
    load (testFile);
    [accuracyMean{i}, accuracyStd{i}, F_LCAMean{i}, FHMean{i}, TIEmean{i}, TestTime{i}] = HierKNNPredictionBatch(data_array, tree, feature{i},str1{i});
    [t_r,~]=size(data_array);
    tiemean{i}=TIEmean{i}(2:end,:)/t_r;
end