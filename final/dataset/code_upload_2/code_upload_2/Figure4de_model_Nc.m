%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data Analysis Code
% Manuscript titled: "Plasticity in local interactions and order transitions in flocking birds"
% Authors: Hangjian Ling, Guillam E. Mclvor, Joe Westley, Kasper van der Vaart, Richard T. Vaughan, 
%          Alex Thornton, Nicholas T. Ouellette
% Manuscript submited to "Nature"
% Date 2019/03
% writen by: Hangjian Ling 
%            Stanford University
% email:     linghj@stanford.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data format: 
%     tracks(:,1) = bird id number
%     tracks(:,2:4) = bird position in Cartesian coordinate system
%     tracks(:,5) = time 
%     tracks(:,6:8) = bird velocity 
% Figure 4d-e: 
%      d: group size (N) v.s. order, linear scale at three noise levels
%      e: group size (N)-Nc v.s. order, log scale at three noise levels
%      f: critical number Nc v.s. noise 
% for details of the agent-based model, see 'model\Viscek3D_harmo.m'
% three cases at three noise level, all at one potential strength
%     Case 1: Noise_005.mat, noise=0.05, potential strength=0.04
%     Case 2: Noise_010.mat, noise=0.10, potential strength=0.04
%     Case 3: Noise_015.mat, noise=0.15, potential strength=0.04

%% Figure 4d, N v.s. order at three noise levels
figure('units','inches','position',[2 2 2.5 2.8]);
load('model/Noise_005.mat');
plot(N(3:end),orderN(3:end),'k-','LineWidth',1);hold on
h1=plot(N(3:end),orderN(3:end),'k^','LineWidth',1,'MarkerFacecolor','b','MarkerSize',8);hold on
load('model/Noise_010.mat');
plot(N,orderN,'k-','LineWidth',1);hold on
h2=plot(N,orderN,'ko','LineWidth',1,'MarkerFacecolor','m','MarkerSize',10);hold on
load('model/Noise_015.mat');
plot(N,orderN,'k-','LineWidth',1);hold on
h3=plot(N,orderN,'ks','LineWidth',1,'MarkerFacecolor','c','MarkerSize',10);hold on
axis([0 200 0.2 1]);box off
set(gca,'Position',[0.15 0.15 0.75 0.75]);
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on'); 
legend([h1 h2 h3],{'\eta=0.05','\eta=0.10','\eta=0.15'},'FontSize',11,...
    'edgecolor','none');

%% Figure 4d, N-Nc v.s. order at three noise levels, log scale 
figure('units','inches','position',[2 2 2.5 2.8]);
load('model/Noise_005.mat');
h1=plot(N(3:end),orderN(3:end),'k-^','LineWidth',1,'MarkerFacecolor','b','MarkerSize',10);hold on
x=[4.5 18];
y=[0.6 1];
p=polyfit(log(x),log(y),1)
plot(x,y,'r--','LineWidth',1.5);hold on

load('model/Noise_010.mat');
h2=plot(N-25.5,orderN,'k-o','LineWidth',1,'MarkerFacecolor','m','MarkerSize',10);hold on
x=[3.3 40];
y=[0.4 1];
p=polyfit(log(x),log(y),1)
plot(x,y,'r--','LineWidth',1.5);hold on

load('model/Noise_015.mat');
h3=plot(N-100,orderN,'k-s','LineWidth',1,'MarkerFacecolor','c','MarkerSize',10);hold on
x=[9 105];
y=[0.4 1];
p=polyfit(log(x),log(y),1)
plot(x,y,'r--','LineWidth',1.5);hold on

axis([3 150 0.4 1]);box off
set(gca,'Position',[0.15 0.15 0.75 0.75]);
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on',...
    'xtick',[5 10 20 100],...
    'xscale','log','yscale','log'); 

%%%% plot critial density v.s. noise
figure('units','inches','position',[2 5 2.5 2.8]);
Nc=[0 25 100];
Er=[1 10 20];
Noise=[0.05 0.1 0.15];
errorbar(Noise,Nc,Er,'k-o','LineWidth',1,'MarkerFacecolor','y','MarkerSize',10);hold on
axis([0.03 0.2 0 130]);box off
set(gca,'Position',[0.18 0.15 0.75 0.75]);
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on'); 
