
data_num = 100;
dataset_num = 10;  
M = 3;
r = 1;
Data = ones(dataset_num,data_num,M)*nan;
HVCval = ones(dataset_num,data_num)*nan;
% generate solution set and HVC
seed = 1;
Data = generateTrainingData(M,Data,data_num,dataset_num,seed);
%Data(1,1,1)
rng('shuffle');
for i=1:dataset_num
    data = Data(i,:,:);
    data = data(1,~isnan(data(1,:,1)),:);
    %b=Data(~isnan(a(:,1)),:);
    hvc = CalHVC(data,r); 
    HVCval(i,1:length(hvc)) = CalHVC(data,r);  
end
    
save('testdata-HVC.mat','Data','HVCval');