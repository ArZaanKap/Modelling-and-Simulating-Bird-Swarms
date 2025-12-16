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
% Figure 4: 
%      a-b: agent-based model results at two group density levels (low and high)
%        c: time-variation of group order for model flocks
% for details of the agent-based model, see 'model\Viscek3D_harmo.m'

%% Figure 4a, trajectories at low density
clear all;clc
load('model/Sample_N10.mat'); 
NumBird=unique(track(:,1));
for j=1:length(NumBird)
    C{j}.color=rand(1,3); % color
end
t=unique(track(:,5));
%%%%% calculate the time-dependent variables
for i=1:length(t)
    %%% select the birds at this time
    id=find(track(:,5)==t(i));
    xyz=track(id,2:4);
    u=track(id,6:8);
    %%%% group order
    p(i)=1/size(u,1)*sum((sum(u./sum(u.^2,2).^0.5,1)).^2)^0.5;
    %%%% group density
    D_temp=[];
    for j=1:size(xyz,1)
        Dist_rank=sum((xyz-xyz(j,:)).^2,2).^0.5;        
        D_temp=[D_temp;max(Dist_rank)];
    end
    D(i)=length(NumBird)/mean(D_temp)^3*6/pi;
end
save('model/sample_lowOrder.mat','t','p','D');

figure('units','inches','position',[5 2 10 8]);
%%% create a projection plane on the horizontal plane
Pj_z=-30; 
X = [min(track(:,2))-1 max(track(:,2))+1 max(track(:,2))+1 min(track(:,2))-1];
Y = [min(track(:,3))-1 min(track(:,3))-1 max(track(:,3))+1 max(track(:,3))+1];
Z = [Pj_z-0.1 Pj_z-0.1 Pj_z-0.1 Pj_z-0.1];
fill3(X,Y,Z,[0.8 0.8 0.8],'edgecolor','none');hold on %%% create a projection plane in xy
%%% plot 3D trajectories
for i=1:length(NumBird)
    id=find(track(:,1)==NumBird(i) & track(:,5)<t(end));
    xyz=track(id,2:4);
    u=track(id,6:8);
    plot3(xyz(:,1),xyz(:,2),xyz(:,3),'-','color',C{i}.color,'lineWidth',3);hold on 
    scatter3(xyz(end,1),xyz(end,2),xyz(end,3),150,[0.2 0.2 0.2],'o','filled');
    
    plot3(xyz(:,1),xyz(:,2),repmat(Pj_z,[size(xyz,1),1]),'-','color',[0.5 0.5 0.5],'lineWidth',2.5);hold on 
    scatter3(xyz(end,1),xyz(end,2),Pj_z,100,[0.2 0.2 0.2],'o','filled');
end
axis equal;grid off;box on;axis off
set(gca,'Position',[0.05 0.1 0.9 0.85])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on','boxstyle','full'); 
plot3([X(1) X(1)+10],[Y(1) Y(1)],[Z(1) Z(1)],'k-','lineWidth',6);
view([45 30]) 

%% Figure 4a, trajectories at high density
clear all;clc
load('model/Sample_N50.mat'); 
NumBird=unique(track(:,1));
for j=1:length(NumBird)
    C{j}.color=rand(1,3); % color
end
t=unique(track(:,5));
%%%%% calculate the time-dependent variables
for i=1:length(t)
    %%% select the birds at this time
    id=find(track(:,5)==t(i));
    xyz=track(id,2:4);
    u=track(id,6:8);
    %%%% group order
    p(i)=1/size(u,1)*sum((sum(u./sum(u.^2,2).^0.5,1)).^2)^0.5;
    %%%% group density
    D_temp=[];
    for j=1:size(xyz,1)
        Dist_rank=sum((xyz-xyz(j,:)).^2,2).^0.5;        
        D_temp=[D_temp;max(Dist_rank)];
    end
    D(i)=length(NumBird)/mean(D_temp)^3*6/pi;
end
save('model/sample_highOrder.mat','t','p','D');

figure('units','inches','position',[5 2 10 8]);
%%% create a projection plane on the horizontal plane
Pj_z=-30; 
X = [min(track(:,2))-1 max(track(:,2))+1 max(track(:,2))+1 min(track(:,2))-1];
Y = [min(track(:,3))-1 min(track(:,3))-1 max(track(:,3))+1 max(track(:,3))+1];
Z = [Pj_z-0.1 Pj_z-0.1 Pj_z-0.1 Pj_z-0.1];
fill3(X,Y,Z,[0.8 0.8 0.8],'edgecolor','none');hold on %%% create a projection plane in xy
%%% plot 3D trajectories
for i=1:length(NumBird)
    id=find(track(:,1)==NumBird(i) & track(:,5)<t(end));
    xyz=track(id,2:4);
    u=track(id,6:8);
    plot3(xyz(:,1),xyz(:,2),xyz(:,3),'-','color',C{i}.color,'lineWidth',3);hold on 
    scatter3(xyz(end,1),xyz(end,2),xyz(end,3),150,[0.2 0.2 0.2],'o','filled');
    
    plot3(xyz(:,1),xyz(:,2),repmat(Pj_z,[size(xyz,1),1]),'-','color',[0.5 0.5 0.5],'lineWidth',2.5);hold on 
    scatter3(xyz(end,1),xyz(end,2),Pj_z,100,[0.2 0.2 0.2],'o','filled');
end
axis equal;grid off;box on;axis off
set(gca,'Position',[0.05 0.1 0.9 0.85])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on','boxstyle','full'); 
plot3([X(1) X(1)+10],[Y(1) Y(1)],[Z(1) Z(1)],'k-','lineWidth',6);
view([45 30]) 

%% Figure 4c, plot time-dependent order for low and high densities
figure('units','inches','position',[5 2 2.5 2.8]);
load('model/sample_lowOrder.mat');
h1=plot(t-min(t),p,'--','LineWidth',1.5,'color',[0.5 0.5 0.5]);hold on
load('model/sample_highOrder.mat');
h2=plot(t-min(t),p,'-','LineWidth',1.5,'color','r');hold on
axis([0 5 0.05 1.0]);box off
set(gca,'Position',[0.18 0.15 0.75 0.75])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on','yscale','linear'); 
legend([h1 h2],{'a','b'},'FontSize',11,'Orientation','horizontal','FontWeight','Bold')

