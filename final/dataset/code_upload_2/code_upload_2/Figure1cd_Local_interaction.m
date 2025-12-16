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
% Figure 1cd: calculate the local interactions in mobbing and transit
% flocks includeing local alignment angle, and neighbour structures
%    r: 1st neighbour position relative to focal birds, focal birds located at (0,0,0)
%    Ang: alignment angle between focal bird with neighbour position

%% start
clear all;clc
%%%%%%%%%%%%%%%%%%%%%%%%%% collect data for mobbing flocks %%%%%%%%%%%%%%%
%%%% obtain data by runing code 'cal_interaction_mob.m' first
r_mob=[]; % 1st neighbour position for mobbing flocks
Ang_mob=[]; % alignment angle for mobbing flocks
load('data/mob_interaction_01.mat');r_mob=[r_mob;r];Ang_mob=[Ang_mob;Ang];
load('data/mob_interaction_02.mat');r_mob=[r_mob;r];Ang_mob=[Ang_mob;Ang];
load('data/mob_interaction_03.mat');r_mob=[r_mob;r];Ang_mob=[Ang_mob;Ang];
load('data/mob_interaction_04.mat');r_mob=[r_mob;r];Ang_mob=[Ang_mob;Ang];
load('data/mob_interaction_05.mat');r_mob=[r_mob;r];Ang_mob=[Ang_mob;Ang];
D_mob=sum(r_mob.^2,2).^0.5; % distance 
D_xy_mob=sum(r_mob(:,1:2).^2,2).^0.5; % distance in horizontal plane

r_mob_n2=[]; % 2st neighbour position for mobbing flocks
Ang_mob_n2=[]; % alignment angle for mobbing flocks
load('data/mob_interaction_01_n2.mat');r_mob_n2=[r_mob_n2;r];Ang_mob_n2=[Ang_mob_n2;Ang];
load('data/mob_interaction_02_n2.mat');r_mob_n2=[r_mob_n2;r];Ang_mob_n2=[Ang_mob_n2;Ang];
load('data/mob_interaction_03_n2.mat');r_mob_n2=[r_mob_n2;r];Ang_mob_n2=[Ang_mob_n2;Ang];
load('data/mob_interaction_04_n2.mat');r_mob_n2=[r_mob_n2;r];Ang_mob_n2=[Ang_mob_n2;Ang];
load('data/mob_interaction_05_n2.mat');r_mob_n2=[r_mob_n2;r];Ang_mob_n2=[Ang_mob_n2;Ang];
load('data/mob_interaction_06_n2.mat');r_mob_n2=[r_mob_n2;r];Ang_mob_n2=[Ang_mob_n2;Ang];
D_mob_n2=sum(r_mob_n2.^2,2).^0.5; % distance 
D_xy_mob_n2=sum(r_mob_n2(:,1:2).^2,2).^0.5; % distance in horizontal plane

r_mob_n3=[]; % 2st neighbour position for mobbing flocks
Ang_mob_n3=[]; % alignment angle for mobbing flocks
load('data/mob_interaction_01_n3.mat');r_mob_n3=[r_mob_n3;r];Ang_mob_n3=[Ang_mob_n3;Ang];
load('data/mob_interaction_02_n3.mat');r_mob_n3=[r_mob_n3;r];Ang_mob_n3=[Ang_mob_n3;Ang];
load('data/mob_interaction_03_n3.mat');r_mob_n3=[r_mob_n3;r];Ang_mob_n3=[Ang_mob_n3;Ang];
load('data/mob_interaction_04_n3.mat');r_mob_n3=[r_mob_n3;r];Ang_mob_n3=[Ang_mob_n3;Ang];
load('data/mob_interaction_05_n3.mat');r_mob_n3=[r_mob_n3;r];Ang_mob_n3=[Ang_mob_n3;Ang];
load('data/mob_interaction_06_n3.mat');r_mob_n3=[r_mob_n3;r];Ang_mob_n3=[Ang_mob_n3;Ang];
D_mob_n3=sum(r_mob_n3.^2,2).^0.5; % distance 
D_xy_mob_n3=sum(r_mob_n3(:,1:2).^2,2).^0.5; % distance in horizontal plane

r_mob_n4=[]; % 2st neighbour position for mobbing flocks
Ang_mob_n4=[]; % alignment angle for mobbing flocks
load('data/mob_interaction_01_n4.mat');r_mob_n4=[r_mob_n4;r];Ang_mob_n4=[Ang_mob_n4;Ang];
load('data/mob_interaction_02_n4.mat');r_mob_n4=[r_mob_n4;r];Ang_mob_n4=[Ang_mob_n4;Ang];
load('data/mob_interaction_03_n4.mat');r_mob_n4=[r_mob_n4;r];Ang_mob_n4=[Ang_mob_n4;Ang];
load('data/mob_interaction_04_n4.mat');r_mob_n4=[r_mob_n4;r];Ang_mob_n4=[Ang_mob_n4;Ang];
load('data/mob_interaction_05_n4.mat');r_mob_n4=[r_mob_n4;r];Ang_mob_n4=[Ang_mob_n4;Ang];
load('data/mob_interaction_06_n4.mat');r_mob_n4=[r_mob_n4;r];Ang_mob_n4=[Ang_mob_n4;Ang];
D_mob_n4=sum(r_mob_n4.^2,2).^0.5; % distance 
D_xy_mob_n4=sum(r_mob_n4(:,1:2).^2,2).^0.5; % distance in horizontal plane

r_mob_n5=[]; % 2st neighbour position for mobbing flocks
Ang_mob_n5=[]; % alignment angle for mobbing flocks
load('data/mob_interaction_01_n5.mat');r_mob_n5=[r_mob_n5;r];Ang_mob_n5=[Ang_mob_n5;Ang];
load('data/mob_interaction_02_n5.mat');r_mob_n5=[r_mob_n5;r];Ang_mob_n5=[Ang_mob_n5;Ang];
load('data/mob_interaction_03_n5.mat');r_mob_n5=[r_mob_n5;r];Ang_mob_n5=[Ang_mob_n5;Ang];
load('data/mob_interaction_04_n5.mat');r_mob_n5=[r_mob_n5;r];Ang_mob_n5=[Ang_mob_n5;Ang];
load('data/mob_interaction_05_n5.mat');r_mob_n5=[r_mob_n5;r];Ang_mob_n5=[Ang_mob_n5;Ang];
load('data/mob_interaction_06_n5.mat');r_mob_n5=[r_mob_n5;r];Ang_mob_n5=[Ang_mob_n5;Ang];
D_mob_n5=sum(r_mob_n5.^2,2).^0.5; % distance 
D_xy_mob_n5=sum(r_mob_n5(:,1:2).^2,2).^0.5; % distance in horizontal plane

% %%%%%%%%%%%%%%%%%%%%%%%%%% collect data for transit flocks %%%%%%%%%%%%%%%
% %%%% obtain data by runing code 'cal_interaction_transit.m' first
r_tra=[]; % 1st neighbour position for transit flocks
Ang_tra=[]; % alignment angle for transit flocks
load('data/transit_interaction_01.mat');r_tra=[r_tra;r];Ang_tra=[Ang_tra;Ang];
load('data/transit_interaction_02.mat');r_tra=[r_tra;r];Ang_tra=[Ang_tra;Ang];
load('data/transit_interaction_03.mat');r_tra=[r_tra;r];Ang_tra=[Ang_tra;Ang];
load('data/transit_interaction_04.mat');r_tra=[r_tra;r];Ang_tra=[Ang_tra;Ang];
load('data/transit_interaction_05.mat');r_tra=[r_tra;r];Ang_tra=[Ang_tra;Ang];
load('data/transit_interaction_06.mat');r_tra=[r_tra;r];Ang_tra=[Ang_tra;Ang];
D_tra=sum(r_tra.^2,2).^0.5; % distance 
D_xy_tra=sum(r_tra(:,1:2).^2,2).^0.5; % distance in horizontal plane

r_tra_n2=[]; % 1st neighbour position for transit flocks
Ang_tra_n2=[]; % alignment angle for transit flocks
load('data/transit_interaction_01_n2.mat');r_tra_n2=[r_tra_n2;r];Ang_tra_n2=[Ang_tra_n2;Ang];
load('data/transit_interaction_02_n2.mat');r_tra_n2=[r_tra_n2;r];Ang_tra_n2=[Ang_tra_n2;Ang];
load('data/transit_interaction_03_n2.mat');r_tra_n2=[r_tra_n2;r];Ang_tra_n2=[Ang_tra_n2;Ang];
load('data/transit_interaction_04_n2.mat');r_tra_n2=[r_tra_n2;r];Ang_tra_n2=[Ang_tra_n2;Ang];
load('data/transit_interaction_05_n2.mat');r_tra_n2=[r_tra_n2;r];Ang_tra_n2=[Ang_tra_n2;Ang];
load('data/transit_interaction_06_n2.mat');r_tra_n2=[r_tra_n2;r];Ang_tra_n2=[Ang_tra_n2;Ang];
D_tra_n2=sum(r_tra_n2.^2,2).^0.5; % distance 
D_xy_tra_n2=sum(r_tra_n2(:,1:2).^2,2).^0.5; % distance in horizontal plane

r_tra_n3=[]; % 1st neighbour position for transit flocks
Ang_tra_n3=[]; % alignment angle for transit flocks
load('data/transit_interaction_01_n3.mat');r_tra_n3=[r_tra_n3;r];Ang_tra_n3=[Ang_tra_n3;Ang];
load('data/transit_interaction_02_n3.mat');r_tra_n3=[r_tra_n3;r];Ang_tra_n3=[Ang_tra_n3;Ang];
load('data/transit_interaction_03_n3.mat');r_tra_n3=[r_tra_n3;r];Ang_tra_n3=[Ang_tra_n3;Ang];
load('data/transit_interaction_04_n3.mat');r_tra_n3=[r_tra_n3;r];Ang_tra_n3=[Ang_tra_n3;Ang];
load('data/transit_interaction_05_n3.mat');r_tra_n3=[r_tra_n3;r];Ang_tra_n3=[Ang_tra_n3;Ang];
load('data/transit_interaction_06_n3.mat');r_tra_n3=[r_tra_n3;r];Ang_tra_n3=[Ang_tra_n3;Ang];
D_tra_n3=sum(r_tra_n3.^2,2).^0.5; % distance 
D_xy_tra_n3=sum(r_tra_n3(:,1:2).^2,2).^0.5; % distance in horizontal plane

r_tra_n4=[]; % 1st neighbour position for transit flocks
Ang_tra_n4=[]; % alignment angle for transit flocks
load('data/transit_interaction_01_n4.mat');r_tra_n4=[r_tra_n4;r];Ang_tra_n4=[Ang_tra_n4;Ang];
load('data/transit_interaction_02_n4.mat');r_tra_n4=[r_tra_n4;r];Ang_tra_n4=[Ang_tra_n4;Ang];
load('data/transit_interaction_03_n4.mat');r_tra_n4=[r_tra_n4;r];Ang_tra_n4=[Ang_tra_n4;Ang];
load('data/transit_interaction_04_n4.mat');r_tra_n4=[r_tra_n4;r];Ang_tra_n4=[Ang_tra_n4;Ang];
load('data/transit_interaction_05_n4.mat');r_tra_n4=[r_tra_n4;r];Ang_tra_n4=[Ang_tra_n4;Ang];
load('data/transit_interaction_06_n4.mat');r_tra_n4=[r_tra_n4;r];Ang_tra_n4=[Ang_tra_n4;Ang];
D_tra_n4=sum(r_tra_n4.^2,2).^0.5; % distance 
D_xy_tra_n4=sum(r_tra_n4(:,1:2).^2,2).^0.5; % distance in horizontal plane

r_tra_n5=[]; % 1st neighbour position for transit flocks
Ang_tra_n5=[]; % alignment angle for transit flocks
load('data/transit_interaction_01_n5.mat');r_tra_n5=[r_tra_n5;r];Ang_tra_n5=[Ang_tra_n5;Ang];
load('data/transit_interaction_02_n5.mat');r_tra_n5=[r_tra_n5;r];Ang_tra_n5=[Ang_tra_n5;Ang];
load('data/transit_interaction_03_n5.mat');r_tra_n5=[r_tra_n5;r];Ang_tra_n5=[Ang_tra_n5;Ang];
load('data/transit_interaction_04_n5.mat');r_tra_n5=[r_tra_n5;r];Ang_tra_n5=[Ang_tra_n5;Ang];
load('data/transit_interaction_05_n5.mat');r_tra_n5=[r_tra_n5;r];Ang_tra_n5=[Ang_tra_n5;Ang];
load('data/transit_interaction_06_n5.mat');r_tra_n5=[r_tra_n5;r];Ang_tra_n5=[Ang_tra_n5;Ang];
D_tra_n5=sum(r_tra_n5.^2,2).^0.5; % distance 
D_xy_tra_n5=sum(r_tra_n5(:,1:2).^2,2).^0.5; % distance in horizontal plane

%% Figure 1c: plot Alignment angle v.s. distance 
edge=0.6:0.8:5.5;
for i=1:length(edge)-2
    id=find(D_tra>edge(i) & D_tra<edge(i+2)); % & D_xy_win>0.2*D_win
    x_win(i)=mean(D_tra(id));
    y_win(i)=median(Ang_tra(id));
    err_win(i)=std(Ang_tra(id))/length(id)^0.5;
    
    id=find(D_tra_n2>edge(i) & D_tra_n2<edge(i+2)); % & D_xy_win>0.2*D_win
    x_win_n2(i)=mean(D_tra_n2(id));
    y_win_n2(i)=median(Ang_tra_n2(id));
    err_win_n2(i)=std(Ang_tra_n2(id))/length(id)^0.5;

    id=find(D_tra_n3>edge(i) & D_tra_n3<edge(i+2)); % & D_xy_win>0.2*D_win
    x_win_n3(i)=mean(D_tra_n3(id));
    y_win_n3(i)=median(Ang_tra_n3(id));
    err_win_n3(i)=std(Ang_tra_n3(id))/length(id)^0.5;

    id=find(D_tra_n4>edge(i) & D_tra_n4<edge(i+2)); % & D_xy_win>0.2*D_win
    x_win_n4(i)=mean(D_tra_n4(id));
    y_win_n4(i)=median(Ang_tra_n4(id));
    err_win_n4(i)=std(Ang_tra_n4(id))/length(id)^0.5;

    id=find(D_tra_n5>edge(i) & D_tra_n5<edge(i+2)); % & D_xy_win>0.2*D_win
    x_win_n5(i)=mean(D_tra_n5(id));
    y_win_n5(i)=median(Ang_tra_n5(id));
    err_win_n5(i)=std(Ang_tra_n5(id))/length(id)^0.5;
  
    id=find(D_mob>edge(i) & D_mob<edge(i+2));% & D_xy_win>0.2*D_win
    x_sum(i)=mean(D_mob(id));
    y_sum(i)=median(Ang_mob(id));
    err_sum(i)=std(Ang_mob(id))/length(id)^0.5;
    
    id=find(D_mob_n2>edge(i) & D_mob_n2<edge(i+2));% & D_xy_win>0.2*D_win
    x_sum_n2(i)=mean(D_mob_n2(id));
    y_sum_n2(i)=median(Ang_mob_n2(id));
    err_sum_n2(i)=std(Ang_mob_n2(id))/length(id)^0.5;

    id=find(D_mob_n3>edge(i) & D_mob_n3<edge(i+2));% & D_xy_win>0.2*D_win
    x_sum_n3(i)=mean(D_mob_n3(id));
    y_sum_n3(i)=median(Ang_mob_n3(id));
    err_sum_n3(i)=std(Ang_mob_n3(id))/length(id)^0.5;
    
    id=find(D_mob_n4>edge(i) & D_mob_n4<edge(i+2));% & D_xy_win>0.2*D_win
    x_sum_n4(i)=mean(D_mob_n4(id));
    y_sum_n4(i)=median(Ang_mob_n4(id));
    err_sum_n4(i)=std(Ang_mob_n4(id))/length(id)^0.5;

    id=find(D_mob_n5>edge(i) & D_mob_n5<edge(i+2));% & D_xy_win>0.2*D_win
    x_sum_n5(i)=mean(D_mob_n5(id));
    y_sum_n5(i)=median(Ang_mob_n5(id));
    err_sum_n5(i)=std(Ang_mob_n5(id))/length(id)^0.5;    
end

figure('units','inches','position',[2 6 3 2.5]);
% h1=plot(x_win,y_win,'k-^','MarkerFaceColor',[1 0.9 1],'MarkerSize',8);hold on
% h2=plot(x_win_n2,y_win_n2,'k-v','MarkerFaceColor',[1 0.7 1],'MarkerSize',8);hold on
% h3=plot(x_win_n3,y_win_n3,'k-s','MarkerFaceColor',[1 0.5 1],'MarkerSize',8);hold on
% h4=plot(x_win_n4,y_win_n4,'k-p','MarkerFaceColor',[1 0.3 1],'MarkerSize',8);hold on
% h5=plot(x_win_n5,y_win_n5,'k-h','MarkerFaceColor',[1 0.1 1],'MarkerSize',8);hold on
% 
% h1=plot(x_sum,y_sum,'k-^','MarkerFaceColor',[0.9 1 1],'MarkerSize',8);hold on
% h2=plot(x_sum_n2,y_sum_n2,'k-v','MarkerFaceColor',[0.7 1 1],'MarkerSize',8);hold on
% h3=plot(x_sum_n3,y_sum_n3,'k-s','MarkerFaceColor',[0.5 1 1],'MarkerSize',8);hold on
% h4=plot(x_sum_n4,y_sum_n4,'k-p','MarkerFaceColor',[0.3 1 1],'MarkerSize',8);hold on
% h5=plot(x_sum_n5,y_sum_n5,'k-h','MarkerFaceColor',[0.1 1 1],'MarkerSize',8);hold on

h1=errorbar(x_win,y_win,err_win*60^0.5*0.2^0.5,'k-^','MarkerFaceColor',[1 0.9 1],'MarkerSize',8);hold on
h2=errorbar(x_win_n2,y_win_n2,err_win_n2*60^0.5*0.2^0.5,'k-v','MarkerFaceColor',[1 0.7 1],'MarkerSize',8);hold on
h3=errorbar(x_win_n3,y_win_n3,err_win_n3*60^0.5*0.2^0.5,'k-s','MarkerFaceColor',[1 0.5 1],'MarkerSize',8);hold on
h4=errorbar(x_win_n4,y_win_n4,err_win_n4*60^0.5*0.2^0.5,'k-p','MarkerFaceColor',[1 0.3 1],'MarkerSize',8);hold on
h5=errorbar(x_win_n5,y_win_n5,err_win_n5*60^0.5*0.2^0.5,'k-h','MarkerFaceColor',[1 0.1 1],'MarkerSize',8);hold on

h1=errorbar(x_sum,y_sum,err_sum*60^0.5*0.2^0.5,'k-^','MarkerFaceColor',[0.9 1 1],'MarkerSize',8);hold on
h2=errorbar(x_sum_n2,y_sum_n2,err_sum_n2*60^0.5*0.2^0.5,'k-v','MarkerFaceColor',[0.7 1 1],'MarkerSize',8);hold on
h3=errorbar(x_sum_n3,y_sum_n3,err_sum_n3*60^0.5*0.2^0.5,'k-s','MarkerFaceColor',[0.5 1 1],'MarkerSize',8);hold on
h4=errorbar(x_sum_n4,y_sum_n4,err_sum_n4*60^0.5*0.2^0.5,'k-p','MarkerFaceColor',[0.3 1 1],'MarkerSize',8);hold on
h5=errorbar(x_sum_n5,y_sum_n5,err_sum_n5*60^0.5*0.2^0.5,'k-h','MarkerFaceColor',[0.1 1 1],'MarkerSize',8);hold on

axis([1 5 0 30]);box off
set(gca,'Position',[0.15 0.15 0.78 0.78])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on');
% legend([h1 h2],{'transit flocks','mobbing flocks'},'FontSize',12,...
%     'edgecolor','none')


%% plot anistropic factor in horizontal plane v.s. distance
edge=0.6:0.8:5.5;
uij=r_tra(:,1:2)./D_xy_tra;
uij_n2=r_tra_n2(:,1:2)./D_xy_tra_n2;
uij_n3=r_tra_n3(:,1:2)./D_xy_tra_n3;
uij_n4=r_tra_n4(:,1:2)./D_xy_tra_n4;
uij_n5=r_tra_n5(:,1:2)./D_xy_tra_n5;
for i=1:length(edge)-2
    id=find(D_tra>edge(i) & D_tra<edge(i+2) & D_xy_tra>0.8*D_tra); 
    x_win(i)=mean(D_tra(id));
    y_win(i)=median(uij(id,2).*uij(id,2)-uij(id,1).*uij(id,1));
    
    id=find(D_tra_n2>edge(i) & D_tra_n2<edge(i+2) & D_xy_tra_n2>0.8*D_tra_n2); 
    x_win_n2(i)=mean(D_tra_n2(id));
    y_win_n2(i)=median(uij_n2(id,2).*uij_n2(id,2)-uij_n2(id,1).*uij_n2(id,1));

    id=find(D_tra_n3>edge(i) & D_tra_n3<edge(i+2) & D_xy_tra_n3>0.6*D_tra_n3); 
    x_win_n3(i)=mean(D_tra_n3(id));
    y_win_n3(i)=median(uij_n3(id,2).*uij_n3(id,2)-uij_n3(id,1).*uij_n3(id,1));
    
    id=find(D_tra_n4>edge(i) & D_tra_n4<edge(i+2) & D_xy_tra_n4>0.6*D_tra_n4); 
    x_win_n4(i)=mean(D_tra_n4(id));
    y_win_n4(i)=median(uij_n4(id,2).*uij_n4(id,2)-uij_n4(id,1).*uij_n4(id,1));
    
    id=find(D_tra_n5>edge(i) & D_tra_n5<edge(i+2) & D_xy_tra_n5>0.6*D_tra_n5); 
    x_win_n5(i)=mean(D_tra_n5(id));
    y_win_n5(i)=median(uij_n5(id,2).*uij_n5(id,2)-uij_n5(id,1).*uij_n5(id,1));

end

uij=r_mob(:,1:2)./D_xy_mob;
uij_n2=r_mob_n2(:,1:2)./D_xy_mob_n2;
uij_n3=r_mob_n3(:,1:2)./D_xy_mob_n3;
uij_n4=r_mob_n4(:,1:2)./D_xy_mob_n4;
uij_n5=r_mob_n5(:,1:2)./D_xy_mob_n5;
for i=1:length(edge)-2
    id=find(D_mob>edge(i) & D_mob<edge(i+2) & D_xy_mob>0.8*D_mob);% & D_xy_win>0.2*D_win
    x_sum(i)=mean(D_mob(id));
    y_sum(i)=median(uij(id,2).*uij(id,2)-uij(id,1).*uij(id,1));
    
    id=find(D_mob_n2>edge(i) & D_mob_n2<edge(i+2) & D_xy_mob_n2>0.8*D_mob_n2);% & D_xy_win>0.2*D_win
    x_sum_n2(i)=mean(D_mob_n2(id));
    y_sum_n2(i)=median(uij_n2(id,2).*uij_n2(id,2)-uij_n2(id,1).*uij_n2(id,1));
    
    id=find(D_mob_n3>edge(i) & D_mob_n3<edge(i+2) & D_xy_mob_n3>0.8*D_mob_n3);% & D_xy_win>0.2*D_win
    x_sum_n3(i)=mean(D_mob_n3(id));
    y_sum_n3(i)=median(uij_n3(id,2).*uij_n3(id,2)-uij_n3(id,1).*uij_n3(id,1));
    
    id=find(D_mob_n4>edge(i) & D_mob_n4<edge(i+2) & D_xy_mob_n4>0.6*D_mob_n4);% & D_xy_win>0.2*D_win
    x_sum_n4(i)=mean(D_mob_n4(id));
    y_sum_n4(i)=median(uij_n4(id,2).*uij_n4(id,2)-uij_n4(id,1).*uij_n4(id,1));
    
    id=find(D_mob_n5>edge(i) & D_mob_n5<edge(i+2) & D_xy_mob_n5>0.8*D_mob_n5);% & D_xy_win>0.2*D_win
    x_sum_n5(i)=mean(D_mob_n5(id));
    y_sum_n5(i)=median(uij_n5(id,2).*uij_n5(id,2)-uij_n5(id,1).*uij_n5(id,1));
end
figure('units','inches','position',[5 2 3 2.5]);
h1=plot(x_win,y_win,'k-^','MarkerFaceColor',[1 0.9 1],'MarkerSize',8);hold on
h2=plot(x_win_n2,y_win_n2,'k-v','MarkerFaceColor',[1 0.7 1],'MarkerSize',8);hold on
h3=plot(x_win_n3,y_win_n3,'k-s','MarkerFaceColor',[1 0.5 1],'MarkerSize',8);hold on
h4=plot(x_win_n4,y_win_n4,'k-p','MarkerFaceColor',[1 0.3 1],'MarkerSize',8);hold on
h5=plot(x_win_n5,y_win_n5,'k-h','MarkerFaceColor',[1 0.1 1],'MarkerSize',8);hold on

plot([1 5],[0 0],'k--');
axis([1 5 -0.05 0.6]);box off
set(gca,'Position',[0.15 0.15 0.78 0.78])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on');
legend([h1 h2 h3 h4 h5],{'n=1','n=2','n=3','n=4','n=5'},'FontSize',12,...
    'edgecolor','none'); % 'Orientation','horizontal'


figure('units','inches','position',[2 2 3 2.5]);
h1=plot(x_sum,y_sum,'k-^','MarkerFaceColor',[0.9 1 1],'MarkerSize',8);hold on
h2=plot(x_sum_n2,y_sum_n2,'k-v','MarkerFaceColor',[0.7 1 1],'MarkerSize',8);hold on
h3=plot(x_sum_n3,y_sum_n3,'k-s','MarkerFaceColor',[0.5 1 1],'MarkerSize',8);hold on
h4=plot(x_sum_n4,y_sum_n4,'k-p','MarkerFaceColor',[0.3 1 1],'MarkerSize',8);hold on
h5=plot(x_sum_n5,y_sum_n5,'k-h','MarkerFaceColor',[0.1 1 1],'MarkerSize',8);hold on

plot([1 5],[0 0],'k--');
axis([1 5 -0.05 0.6]);box off
set(gca,'Position',[0.15 0.15 0.78 0.78])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on');