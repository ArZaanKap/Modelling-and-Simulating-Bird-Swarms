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
% Figure 3: plot group order v.s. group density for transit and mobbing
% flocks, in transit flock, subgroup size = 5 
% Extended Data Fig 2: plot group order v.s. group density for transit and mobbing
% flocks, in transit flock, subgroup size = 10 or 20

%% start
clear all;clc
%%%%%%%%%%%%%%%%%%%%%%%%% load data for 154 mobbing groups %%%%%%%%%%%%%%%
%%% get group order of mobbing flocks by running code 'cal_order_mob.m' 
%%% and see more details in 'Data/Mobbing_flocks.xlsx'
%%% Day 1: event # 106 
density_1=[0.31 1.9 1.8 9.4 14.1 22.7 2.1 4.1 33.2 19.5 0.96 4.1 2.6 0.25]/1000;
density_e1=[0.02 1.3 0.6 2.7 2.7 6.8 0.8 2.3 2.2 10.5 0.31 2 3 0.11]/1000; % standard error
order_1=[0.21 0.35 0.72 0.98 0.97 0.96 0.36 0.97 0.99 0.96 0.74 0.88 0.78 0.54];
order_e1=[0.04 0.2 0.15 0 0.02 0.02 0.12 0.01 0 0.04 0.18 0.14 0.09 0.32]; % standard error
N1=[4 8 10 11 7 7 9 5 8 8 7 5 4 4]; % bird number in group

%%%% Day 2: event # 107 
density_2=[3.26 8.49 7.33 20.3 27.3 1 0.98 0.64 69.7 1.42 71.7 7 5.86 1.19 1.64 1.99 5.22]/1000;
density_e2=[1.44 4.14 3.65 9.44 7.7 0.31 0.37 0.13 17.7 0.44 40.7 1.5 3.9 0.12 0.21 0.55 1.42]/1000;
order_2=[0.65 0.85 0.83 0.95 0.99 0.59 0.53 0.51 0.99 0.51 0.98 0.9 0.84 0.71 0.76 0.57 0.95];
order_e2=[0.28 0.11 0.16 0.05 0.01 0.24 0.12 0.04 0 0.16 0.02 0.1 0.14 0.11 0.04 0.18 0.03];
N2=[6 6 7 5 4 5 4 4 5 4 4 7 4 6 6 4 4]; % bird number in group

%%%% Day 3: event # 108 
density_3=[2.1 3.12 0.67 1.89 1.04 0.51 1.13 5.57 1.1 1.93 1.29 0.94 26.9 23.8 3.17 19.7 25.6 34.3 19.8 20 1.33 0.96 2.13 1.44 1.55 64]/1000;
density_e3=[1 1.14 0.07 0.89 0.43 0.05 0.36 2.36 0.21 0.52 0.38 0.08 7.6 6.2 0.28 3.2 4.2 24.9 4.9 20 0.77 0.2 0.42 0.33 0.42 46]/1000;
order_3=[0.56 0.63 0.57 0.39 0.23 0.22 0.56 0.92 0.73 0.8 0.47 0.43 0.97 0.96 0.93 0.98 0.95 0.95 0.97 0.9 0.42 0.39 0.57 0.43 0.38 0.97];
order_e3=[0.18 0.17 0.11 0.2 0.09 0.04 0.07 0.08 0.06 0.02 0.12 0.17 0.06 0.04 0.01 0.02 0.05 0.05 0.01 0.1 0.12 0.02 0.09 0.15 0.19 0.02];
N3=[5 4 4 8 8 7 9 7 6 7 7 8 4 6 6 5 7 4 7 4 4 7 6 5 4 4];

%%%% Day 4: event # 112 
density_4=[0.26 0.78 10.5 6.6 6 12.5 4.4 15.2 11.2 2.4 21.2 7.4 29.8 21.6 20.1 13 5.5 13.6 23.3 1.41 22.1 4 4.2 1.96 0.54 6.08 7.8 0.84 46.4 7.7]/1000;
density_e4=[0.07 0.08 2.96 1.2 2.2 2.04 6.3 1.71 6.7 0.5 0.51 1.75 6.5 5.4 7.2 4.5 2.1 3.55 10.3 0.56 6.4 0.46 0.28 1.11 0.08 2.32 4.4 0.3 37 7.9]/1000;
order_4=[0.32 0.89 0.81 0.95 0.95 0.93 0.72 0.95 0.97 0.96 0.96 0.98 0.97 0.89 0.93 0.9 0.98 0.97 0.93 0.67 0.96 0.98 0.95 0.62 0.77 0.93 0.91 0.46 0.99 0.92];
order_e4=[0.04 0.04 0.07 0.03 0.04 0.05 0.25 0.06 0.04 0.02 0.04 0.01 0 0.09 0.02 0.06 0.02 0.01 0.05 0.07 0.02 0.02 0.02 0.12 0.04 0.02 0.06 0.27 0 0.04];
N4=[5 6 8 5 7 6 7 10 6 7 20 6 23 7 22 9 4 15 6 15 29 12 16 7 9 13 7 7 4 7];

%%%% Day 5: event # 113 
density_5=[2.1 2.7 1.48 5 2.5 1.44 16.7 11 14.1 16.7 42.2 0.69 1.36 0.59 0.89 0.61 8.6 1.92 1.34]/1000;
density_e5=[1.8 1.3 0.79 0.92 1.8 0.06 8.8 4.8 5.1 0.88 5.8 0.11 1.18 0.12 0.62 0.15 3.8 0.93 0.3]/1000;
order_5=[0.62 0.54 0.22 0.93 0.92 0.54 0.9 0.91 0.94 0.93 0.96 0.52 0.4 0.51 0.6 0.5 0.92 0.93 0.83];
order_e5=[0.12 0.14 0.07 0.06 0.03 0.06 0.08 0.07 0.07 0.04 0.04 0.14 0.18 0.11 0.29 0.26 0.06 0.03 0.15];
N5=[4 5 4 6 6 7 5 11 6 12 6 7 5 6 4 4 4 8 5];

%%%% Day 6: event # 114 
density_6=[0.85 18.2 1.55 1.06 5.5 1.3 2.1 9.8 9.7 2.01 1.59 0.51]/1000;
density_e6=[0.23 19 0.97 0.6 0.18 0.4 0.02 0.75 2.7 0.55 1.55 0.11]/1000;
order_6=[0.39 0.95 0.4 0.6 0.93 0.64 0.99 0.99 0.98 0.93 0.4 0.17];
order_e6=[0.18 0.03 0.12 0.18 0.05 0.27 0 0 0 0.07 0.11 0.07];
N6=[4 4 4 5 5 4 4 7 5 4 4 7];

%%% Day 7: event # 110 c+d
density_7=[0.69 1.7 0.99 0.66 2.33 20 2.89 4.2 1.86 1.11 2.34]/1000;
density_e7=[0.16 0.27 0.29 0.02 0.68 10 1.3 1.2 1.15 0.25 1.26]/1000;
order_7=[0.28 0.97 0.8 0.75 0.77 0.98 0.76 0.92 0.69 0.61 0.82];
order_e7=[0.12 0.02 0.11 0.07 0.14 0.02 0.2 0.04 0.24 0.08 0.1];
N7=[4 4 6 7 3 3 6 4 5 6 3];

%%% Day 8: event # 111 a+b
density_8=[3.6 2.86 11.4 2.2 1.3 6.3 10.7 3.9 1.14 3.28 5 1.8 3.36 68]/1000;
density_e8=[2.4 0.87 2.2 0.42 0.9 1.2 3.9 1.2 0.54 1.55 0.52 0.85 0.24 18]/1000;
order_8=[0.76 0.75 0.92 0.34 0.72 0.73 0.94 0.54 0.21 0.72 0.89 0.71 0.96 0.98];
order_e8=[0.17 0.1 0.04 0.08 0.19 0.12 0.04 0.21 0.05 0.21 0.04 0.28 0.02 0];
N8=[4 3 5 4 3 4 6 8 5 7 5 7 4 7];
 
%%% Day 9: event # 115 a
density_9=[0.42 4.2 86 1.5 1.91 17.9 0.36]/1000;
density_e9=[0.16 2.1 92 0.57 1.69 14.3 0.1]/1000;
order_9=[0.34 0.96 0.98 0.55 0.33 0.93 0.25];
order_e9=[0.07 0.02 0.02 0.05 0.06 0.06 0.08];
N9=[4 3 3 4 3 3 4];

%%% Day 10: event # 118 a
density_10=[2.69 0.58 10.8 0.5]/1000;
density_e10=[0.98 0.04 6.6 0.37]/1000;
order_10=[0.96 0.8 0.88 0.81];
order_e10=[0.03 0.02 0.06 0.16];
N10=[3 4 4 3];

%%% all 154 mobbing flock data
density_mob=[density_1,density_2,density_3,density_4,density_5,density_6,...
    density_7,density_8,density_9,density_10];
order_mob=[order_1,order_2,order_3,order_4,order_5,order_6,...
    order_7,order_8,order_9,order_10];
order_e=[order_e1,order_e2,order_e3,order_e4,order_e5,order_e6,...
    order_e7,order_e8,order_e9,order_e10];
density_e=[density_e1,density_e2,density_e3,density_e4,density_e5,density_e6,...
    density_e7,density_e8,density_e9,density_e10];
Num=[N1,N2,N3,N4,N5,N6,N7,N8,N9,N10];
%%%%%%%%%%%%%%%%%%%%%%%%% Finish load data for 154 mobbing groups %%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% load data for transit flocks %%%%%%%%%%%%%%%%%
%%% get group order of transit flocks by running code 'cal_order_transit.m' 
A1=load('data/transit_order_01.mat'); 
A2=load('data/transit_order_02.mat');
A3=load('data/transit_order_03.mat');
A4=load('data/transit_order_04.mat');
A5=load('data/transit_order_05.mat');
A6=load('data/transit_order_06.mat');
order_tra=[A1.order;A2.order;A3.order;A4.order;A5.order;A6.order];
density_tra=[A1.density;A2.density;A3.density;A4.density;A5.density;A6.density];
%%%%%%%%%%%%%%%%%%%%%%%%% Finish load data for transit flocks  %%%%%%%%%%%%

figure('units','inches','position',[2 2 4 3.5]);
edge=0.0005:0.0008:0.02;
for i=1:length(edge)-2
    id=find(density_mob>edge(i) & density_mob<edge(i+2));
    x1(i)=mean(density_mob(id));
    y1(i)=mean(order_mob(id)); %%% mobbing flocks
    err1(i)=std(order_mob(id))/length(id)^0.5;
    
    id=find(density_tra>edge(i) & density_tra<edge(i+2));
    x2(i)=mean(density_tra(id));
    y2(i)=mean(order_tra(id)); %%% transit flocks
    err2(i)=std(order_tra(id))/length(id)^0.5;
end
errorbar(x1,y1,err1,'k','LineWidth',1.5,'LineStyle','none');hold on
h1=plot(x1,y1,'ko','LineWidth',1,'MarkerFacecolor','c','MarkerSize',10);hold on

h2=plot(x2,y2,'ks','LineWidth',1,'MarkerFacecolor','m','MarkerSize',10);hold on
errorbar(x2,y2,err2,'k','LineWidth',1.5,'LineStyle','none');hold on

x1=[0.0002 0.005];
y1=[0.28 0.92];
p=polyfit(log(x1),log(y1),1) % fitting for the mobbing flock
h3=plot([0.0002:0.0005:0.01],exp(polyval(p,log([0.0002:0.0005:0.01]))),...
    'r--','LineWidth',1.5);

axis([0.0005 0.02 0.45 1]);box off
set(gca,'Position',[0.13 0.13 0.8 0.8]);
set(gca,'FontSize',14,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on',...
    'xtick',[0.001 0.01 0.02],...
    'xscale','log','yscale','log'); 
legend([h2 h1],{'transit flocks','mobbing flocks'},...
    'edgecolor','none','fontsize',14);