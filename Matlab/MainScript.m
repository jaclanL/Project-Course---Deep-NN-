clc
clear variables

%% Initialize PC-ESN
nInputUnits = 21;
nReservoirUnits = 600;
nOutputUnits = 7;
spectralRadius = 0.2;
sigma2 = 50;
phi2 = 1;
outFactor = 0.2; % 0 < x < 1

pcesn = PCESN;
pcesn = initPCESN(pcesn,nInputUnits,nReservoirUnits,nOutputUnits,spectralRadius,sigma2, phi2,outFactor);

%% Load data and normalize input
load('sarcos_inv.mat');

inData = sarcos_inv(:, 1:21);
outData = sarcos_inv(:, 22:end);

inData = normalize_input(inData);

%% Train network
data_len = 3000; % length(inData)
output = zeros(7,data_len);

for i=1:data_len
    pcesn = trainESN(pcesn,inData(i,:)',outData(i,:)');
    output(:,i) = pcesn.o;
    disp(i)
end

pcesn_trained = pcesn;

%% Transfer learning with latest network state
%data_len2 = 100;
%output2 = zeros(7,data_len2);
%for j=1:data_len2
%    pcesn_trained = train(pcesn_trained,inData(i,:)',outData(i,:)');
%    output2(:,j) = pcesn_trained.o;
%    disp(j)
%end

%% Plot true vs output
close all

plot(output(1,2800:3000))
hold on
plot(outData(2800:3000,1))

[e1, e2] = normalized_mse(outData(2500:3000,:),output(:,2500:3000)');
