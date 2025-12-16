%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main function to run the Vicsek Model in 3D with potential well
% Manuscript titled: "Plasticity in local interactions and order transitions in flocking birds"
% Authors: Hangjian Ling, Guillam E. Mclvor, Joe Westley, Kasper van der Vaart, Richard T. Vaughan, 
%          Alex Thornton, Nicholas T. Ouellette
% Manuscript submited to "Nature"
% Date 2019/03
% writen by: Hangjian Ling 
%            Stanford University
% email:     linghj@stanford.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;clc

%% Test case 01: noise=0.05 
folderName='Results_noise0_05/';
EventNum=1:50;
N=round(exp(1.5:0.2:3.4));
noise=0.05;
orderN=[];densityN=[];
for i=1:length(N)
    Viscek3D_harmo(EventNum,N(i),noise);
    [order_temp,density_temp]=cal_order(folderName,N(i));
    orderN(i)=mean(order_temp);
    densityN(i)=mean(density_temp);
end
save('Noise_005.mat','N','densityN','orderN');

%% Test case 02: noise=0.10 
folderName='Results_noise0_10/';
EventNum=1:50;
N=round(exp(2.8:0.2:4.6)); 
noise=0.10;
orderN=[];densityN=[];
for i=1:length(N)
    Viscek3D_harmo(EventNum,N(i),noise);
    [order_temp,density_temp]=cal_order(folderName,N(i));
    orderN(i)=mean(order_temp);
    densityN(i)=mean(density_temp);
end
save('Noise_010.mat','N','densityN','orderN');

%% Test case 03: noise=0.15
folderName='Results_noise0_15/';
EventNum=1:50;
N=round(exp(2.8:0.2:4.6)); 
noise=0.15;
orderN=[];densityN=[];
for i=1:length(N)
    Viscek3D_harmo(EventNum,N(i),noise);
    [order_temp,density_temp]=cal_order(folderName,N(i));
    orderN(i)=mean(order_temp);
    densityN(i)=mean(density_temp);
end
save('Noise_015.mat','N','densityN','orderN');
