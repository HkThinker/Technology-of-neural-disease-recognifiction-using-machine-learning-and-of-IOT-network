%% 清空数据集
warning off
close all
clear 
clc



%% 划分数据集和测试集

%1.不提取特征 res_lstm.mat
%%导入数据集
% load("res_lstm.mat")
% temp=1:1:1917887;
% 
% P_train=res_lstm(temp(1:1726099),1:9)';
% T_train=res_lstm(temp(1:1726099),10)';
% M=size(P_train,2);
% 
% P_test=res_lstm(temp(1726100:end),1:9)';
% T_test=res_lstm(temp(1726100:end),10)';
% N=size(P_test,2);

% 提取特征 feature.mat
load("pd_voice.mat")

temp=randperm(756); % 1086*77   756*754

P_train=pdspeechfeatures(temp(76:end),1:753)';
T_train=pdspeechfeatures(temp(76:end),754)';
M=size(P_train,2);

P_test=pdspeechfeatures(temp(1:75),1:753)';
T_test=pdspeechfeatures(temp(1:75),754)';
N=size(P_test,2);


%% 数据归一化

[P_train,ps_input]=mapminmax(P_train,0,1);
P_test=mapminmax('apply',P_test,ps_input);

t_train=categorical(T_train)';
t_test=categorical(T_test)';

%% 数据平铺
% 将数据平铺成1维数据只是一种处理方式
% 也可以平铺成2维数据,以及3维数据,需要修改对应模型的结构
% 但是应该始终和输入层数据结构保持一致
P_train=double(reshape(P_train,[753,1,1,M]));
P_test=double(reshape(P_test,[753,1,1,N]));

%% 数据格式转换
for i =1 :M
    p_train{i,1}=P_train(:,:,1,i);
end

for i =1 :N
    p_test{i,1}=P_test(:,:,1,i);
end

%% 创建网络
layers=[ ...
    sequenceInputLayer(753)             %输入层
    gruLayer(6,'OutputMode','last')  %LSTM层
    reluLayer                         %Relu激活层
    fullyConnectedLayer(2)            %全连接层
    softmaxLayer                      %分类层
    classificationLayer];
% layers = [
%     sequenceInputLayer(75,"Name","input")
%     gruLayer(6,'OutputMode','last')
%     dropoutLayer(0.5,"Name","dropout")
%     gruLayer(6,'OutputMode','last')
%     dropoutLayer(0.5,"Name","dropout_1")
%     gruLayer(6,'OutputMode','last')
%     dropoutLayer(0.5,"Name","dropout_2")
%     gruLayer(6,'OutputMode','last')
%     dropoutLayer(0.5,"Name","dropout_3")
%     gruLayer(6,'OutputMode','last')
%     dropoutLayer(0.5,"Name","dropout_4")
%     gruLayer(6,'OutputMode','last')
%     dropoutLayer(0.5,"Name","dropout_5")
%     fullyConnectedLayer(2)            %全连接层
%     softmaxLayer("Name","softmax")
%     classificationLayer("Name","classification")];

%参数设置
options=trainingOptions( "adam", ...     %Adam梯度下降算法
    "MiniBatchSize", 64, ...            %批大小
    "MaxEpochs",1000, ...                %最大迭代次数
    "InitialLearnRate",1e-2 , ...        %初始学习率
    "LearnRateSchedule",'piecewise',...  %学习率下降
    'LearnRateDropFactor',0.1,...        %学习率下降因子
    'LearnRateDropPeriod',700,...        %经过700次训练后,学习率为0.01*0.1
    'Shuffle','every-epoch',...          %每次训练打乱数据集
    'ValidationPatience',Inf,...         %关闭验证
    'Plots','training-progress',...      %画出曲线
    'Verbose',false);

%% 训练模型
net=trainNetwork(p_train,t_train,layers,options);

%% 仿真预测
t_sim1=predict(net,p_train);
t_sim2=predict(net,p_test);

%% 数据反归一化
T_sim1=vec2ind(t_sim1')-1;
T_sim2=vec2ind(t_sim2')-1;


%% 性能评估
error1=sum((T_sim1==T_train))/M*100;
error2=sum((T_sim2==T_test))/N*100;

%% 查看网络结构
analyzeNetwork(net)


%% 数据排序
[T_train,index_1]=sort(T_train);
[T_test,index_2]=sort(T_test);

T_sim1=T_sim1(index_1);
T_sim2=T_sim2(index_2);

%% 绘图
figure 
plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1)
legend('True Value','Predicted value')
xlabel('Prediction Sample')
ylabel('Predicted results')
string={'Comparison of prediction results for the training datasets'; ['Accuracy=' num2str(error1) '%']};
title(string)
xlim([1,M])
grid

figure 
plot(1:N,T_test,'r-*',1:N,T_sim2,'b-o','LineWidth',1)
legend('True Value','Predicted value')
xlabel('Prediction Sample')
ylabel('Predicted results')
string={'Comparison of prediction results for the test datasets'; ['Accuracy=' num2str(error2) '%']};
title(string)
xlim([1,N])
grid

%% 混淆矩阵

figure
cm = confusionchart(T_train,T_sim1);
cm.Title =  'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary= 'row-normalized';

figure
cm = confusionchart(T_test,T_sim2);
cm.Title =  'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary= 'row-normalized';







  





