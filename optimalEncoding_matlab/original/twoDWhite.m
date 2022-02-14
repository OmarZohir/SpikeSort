clear;clc;
%% Initializing Learning rates, membrane leak, and timestep
Nneuron= [10 20 30];
Nx=2;       %dimesnion of the input

lambda=50;    %membrane leak
dt=0.001;     %time step

epsr=0.001;  % earning rate of the recurrent connections
epsf=0.0001; %% learning rate of the feedforward connections FF

alpha=0.18; % scaling of the Feefforward weights
beta=1/0.9;  %scaling of the recurrent weights
mu=0.02/0.9; %quadratic cost

%% Learning
%%Initial connectivity
for i = 1:length(Nneuron)

    Fi=0.5*randn(Nx,Nneuron(i)); %the inital feedforward weights are chosen randomely
    Fi=1*(Fi./(sqrt(ones(Nx,1)*(sum(Fi.^2)))));%the FF weights are normalized
    Ci=-0.2*(rand(Nneuron(i),Nneuron(i)))-0.5*eye(Nneuron(i)); %the initial recurrent conectivity is very weak except for the autapses

    Thresh=0.5; %vector of thresholds of the neurons


    %[Fs,Cs,F,C,Decs,ErrorC(i,:)]=Learning(dt,lambda,epsr,epsf,alpha, beta, mu, Nneuron(i),Nx, Thresh,Fi,Ci);
    [~,~,~,~,~,ErrorC(i,:), RMSEDec, MeanPrate(i,:)]=Learning(dt,lambda,epsr,epsf,alpha, beta, mu, Nneuron(i),Nx, Thresh,Fi,Ci);
    RMSE(i,:,:) = RMSEDec;
end
for i = 1:length(Nneuron)
    TotalSpikesByNetwork(i,:) = Nneuron(i)*MeanPrate(i,:);
end
%% Plotting Total Spikes per network vs # of Neurons
figure
loglog((2.^(1:length(TotalSpikesByNetwork)))*dt,TotalSpikesByNetwork');
lgd = legend('10', '20', '30');
title(lgd,'# of Neurons','FontSize',12)
xlabel("Time(s)")
ylabel("Average total Spikes fired by the entire network");
title("Total Number of Spikes vs Size of Network");
saveas(gcf, 'GaussianNormal_TotalSpikes.png');
%% Plotting RMSE at decoder vs # Neurons
figure()
loglog((2.^(1:length(RMSE)))*dt,squeeze(RMSE)');
lgd = legend('10', '20', '30');
title(lgd,'# of Neurons','FontSize',12);
xlabel("Time(s)");
title("RMSE decoder vs Size of Network (Reconstructed signal is smoothened)")
ylabel("RMSE between Input and Reconstructed signal during optimal decoder training");
saveas(gcf, 'GaussianNormal_RMSE_Decoder.png');
%% Plotting RMSE at Testing vs # Neurons
figure()
loglog((2.^(1:length(ErrorC)))*dt,ErrorC');
lgd = legend('10', '20', '30');
title(lgd,'# of Neurons','FontSize',12)
xlabel("Time(s)")
ylabel("RMSE between Input and Reconstructed signal during Running the network using test input");
title("RMSE Test input vs Size of Network (Reconstructed signal is smoothened)");
saveas(gcf, 'GaussianNormal_RMSE_Testing.png');