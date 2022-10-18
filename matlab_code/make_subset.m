% 这个代码用来从1M训练数据中随机选取出1K，10K，100K个数据，用不同的seed控制。
% 作为新的训练数据。

%% load 1M训练数据， only seed 5
load('//10.20.2.245/datasets/HV-Net-datasets/data/train_data_M10_5.mat')
dataset_num = size(Data, 1)
data_num = size(Data, 2)
M = size(Data, 3)

Data_whole = Data; 
HVval_whole = HVval;

%% 随机选取1K数据，用seed 6
rng(6);
selected_indexs = randperm(dataset_num);
selected_indexs = selected_indexs(1:1000);

Data = Data_whole(selected_indexs, :, :);
HVval = HVval_whole(selected_indexs, :, :);

save(['//10.20.2.245/datasets/HV-Net-datasets/data/train_data_M', num2str(M), '_1K_5.mat'],'Data','HVval');

%% 随机选取10K数据，用seed 7
rng(7);
selected_indexs = randperm(dataset_num);
selected_indexs = selected_indexs(1:10000);

Data = Data_whole(selected_indexs, :, :);
HVval = HVval_whole(selected_indexs, :, :);

save(['//10.20.2.245/datasets/HV-Net-datasets/data/train_data_M', num2str(M), '_10K_5.mat'],'Data','HVval');

%% 随机选取100K数据，用seed 8
rng(8);
selected_indexs = randperm(dataset_num);
selected_indexs = selected_indexs(1:100000);

Data = Data_whole(selected_indexs, :, :);
HVval = HVval_whole(selected_indexs, :, :);

save(['//10.20.2.245/datasets/HV-Net-datasets/data/train_data_M', num2str(M), '_100K_5.mat'],'Data','HVval');

