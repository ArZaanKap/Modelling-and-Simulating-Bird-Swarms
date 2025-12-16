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
%     tracks_filt(:,1) = bird id number
%     tracks_filt(:,2:4) = bird position in Cartesian coordinate system
%     tracks_filt(:,5) = time 
%     tracks_filt(:,6:8) = bird velocity 
%     tracks_filt(:,9:11) = bird acceleration
%     tracks_filt(:,12) = bird wingbeat frequency 
% calculate general properties for Table S1: 
%    include time duration, group size, group order, NND, flight speed,
%    flight speed in gravity direction

%% start
clear all;clc
%%%%% for mobbing flocks
load('data/mob_01.mat'); % mob_01, mob_02, mob_03 ...
T=unique(tracks_filt(:,5)); Frames=1:length(T); 

% %%%%% for transit flocks
% load('data/transit_01.mat');Frames=200:650; % transit flock #01
% load('data/transit_02.mat');Frames=750:950; % transit flock #02
% load('data/transit_03.mat');Frames=50:400; % transit flock #03
% load('data/transit_04.mat');Frames=550:900; % transit flock #04
% load('data/transit_05.mat');Frames=100:300; % transit flock #05
% load('data/transit_06.mat');Frames=600:900; % transit flock #06

T=unique(tracks_filt(:,5));
Num=[];% number of birds (group size)
p=[]; % group order
D1=[]; % nearest neighbor distance (NND)
for i=Frames
    id=find(tracks_filt(:,5)==T(i));
    xyz=tracks_filt(id,2:4);
    u=tracks_filt(id,6:8);    
    if size(xyz,1)<4
        continue;
    elseif size(xyz,1)>=4
        p=[p;1/size(u,1)*sum((sum(u./sum(u.^2,2).^0.5,1)).^2)^0.5];
        Num=[Num;size(xyz,1)];    
        D_temp=[];
        for j=1:size(xyz,1)
            D=sum((xyz-xyz(j,:)).^2,2).^0.5;
            [D,I]=sort(D);
            D_temp=[D_temp;D(2)];
        end           
        D1=[D1;mean(D_temp)];
    end       
end
%%% time duration
prop.Time=length(Frames)/60;
%%% group size 
prop.size=mean(Num);
prop.sizeSD=std(Num);
%%% group order 
prop.order=mean(p);
prop.orderSD=std(p);
%%% NND 
prop.NND=mean(D1);
prop.NNDSD=std(D1);
%%% flight speed
u=tracks_filt(:,6:8);
prop.speed=mean(sum(u(:,1:3).^2,2).^0.5);
prop.speedSD=std(sum(u(:,1:3).^2,2).^0.5);
% flight speed in gravity direction
U3=u(:,3); 
prop.speedg=mean(U3);
prop.speedgSD=std(U3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% extra plots
%%%% plot time-dependent varibles
% figure('units','inches','position',[2 8 10 1.5]);
% plot(T,Num,'k-','LineWidth',1); %% number
% box off;axis tight;
% set(gca,'Position',[0.05 0.2 0.9 0.78])
% set(gca,'FontSize',12,'TickLength',[0.02, 0.01],...
%     'XMinorTick','on','YMinorTick','on','BoxStyle','full'); 

%%%%% plot trajectories: 106-108-4000; 112-4000; 113-4800; 114-3200
% curFrame=T(3200);
% Id=find(tracks_filt(:,5)==curFrame);
% NumT=unique(tracks_filt(Id,1));
% for j=1:NumT
%     C{j}.color=rand(1,3); % color
% end
% figure('units','inches','position',[5 2 4 3]);
% for i=1:length(NumT)
%     Id=find(tracks_filt(:,1)==NumT(i) & tracks_filt(:,5)<=curFrame);
%     xyz=tracks_filt(Id,2:4);
%     u=tracks_filt(Id,6:8);
%     U=sum(u.^2,2).^0.5;
%     scatter3(xyz(:,1),xyz(:,2),xyz(:,3),5,C{i}.color,'o','filled');hold on 
%     scatter3(xyz(end,1),xyz(end,2),xyz(end,3),30,'ko','filled');
% end
% axis equal;grid off;box off;
% set(gca,'Position',[0.13 0.13 0.78 0.78])
% set(gca,'FontSize',12,'TickLength',[0.1, 0.05],...
%     'XMinorTick','on','YMinorTick','on','BoxStyle','full'); 
% 