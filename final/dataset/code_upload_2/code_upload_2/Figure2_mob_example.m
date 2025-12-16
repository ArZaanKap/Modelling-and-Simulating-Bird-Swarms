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
% Figure 2: 
%      a-c: mobbing flocks at three group density levels (low, medium, and high)
%        d: time-variation of group order for three mobbing flocks

%% start
clear all;clc
%% figure 2a: mobbing flocks at low density
load('data/mob_03.mat');
id_birds=[62 64 66 72 88 95]; %% 108: good example, density=0.0016
tracks_good=selectBirds(tracks_filt,id_birds);
t=unique(tracks_good(:,5));
%%%%% calculate the time-dependent group order and density
for i=1:length(t)
    %%% select the birds at this time
    id=find(tracks_good(:,5)==t(i));
    xyz=tracks_good(id,2:4);
    u=tracks_good(id,6:8);
    %%%% group order
    p(i)=1/size(u,1)*sum((sum(u./sum(u.^2,2).^0.5,1)).^2)^0.5;
    %%%% group density
    D_temp=[];
    for j=1:size(xyz,1)
        Dist_rank=sum((xyz-xyz(j,:)).^2,2).^0.5;        
        D_temp=[D_temp;max(Dist_rank)];
    end
    D(i)=length(id_birds)/mean(D_temp)^3*6/pi;
end
N=length(id_birds);
save('data/mob_sample_lowOrder.mat','t','p','D','N');
% assign one color for each trajectory
for j=1:length(id_birds)
    C{j}.color=rand(1,3); % color
end
offset=0.3;
figure('units','inches','position',[5 2 10 8]);
%%% create a projection plane on the horizontal plane
Pj_z=-20; 
X = [min(tracks_good(:,2))-1 max(tracks_good(:,2))+1 max(tracks_good(:,2))+1 min(tracks_good(:,2))-1];
Y = [min(tracks_good(:,3))-1 min(tracks_good(:,3))-1 max(tracks_good(:,3))+1 max(tracks_good(:,3))+1];
Z = [Pj_z-offset Pj_z-offset Pj_z-offset Pj_z-offset];
fill3(X,Y,Z,[0.8 0.8 0.8],'edgecolor','none');hold on 
plot3([X(1) X(1)],[Y(3) Y(3)-10],[Z(1) Z(1)],'k-','lineWidth',10); % scale bar
%%% plot 3D trajectories
for i=1:length(id_birds)
    id=find(tracks_good(:,1)==id_birds(i));
    xyz=tracks_good(id,2:4);
    u=tracks_good(id,6:8);
    plot3(xyz(:,1),xyz(:,2),xyz(:,3),'-','color',C{i}.color,'lineWidth',5);hold on 
    scatter3(xyz(end,1),xyz(end,2),xyz(end,3),300,[0.2 0.2 0.2],'o','filled');
    
    plot3(xyz(:,1),xyz(:,2),repmat(Pj_z,[size(xyz,1),1]),'-','color',[0.5 0.5 0.5],'lineWidth',3);hold on 
    scatter3(xyz(end,1),xyz(end,2),Pj_z,150,[0.2 0.2 0.2],'o','filled');
end
axis equal;grid off;box on;axis off
set(gca,'Position',[0.05 0.1 0.9 0.85])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on','boxstyle','full'); 
view([-45 15]) 


%% figure 2b: mobbing flocks at medium density
load('data/mob_06.mat');
id_birds=[1188 1187 1189 1191 1192 1194 1190]; %% 112, good
tracks_good=selectBirds(tracks_filt,id_birds);
t=unique(tracks_good(:,5));
%%%%% calculate the time-dependent group order and density
for i=1:length(t)
    %%% select the birds at this time
    id=find(tracks_good(:,5)==t(i));
    xyz=tracks_good(id,2:4);
    u=tracks_good(id,6:8);
    %%%% group order
    p(i)=1/size(u,1)*sum((sum(u./sum(u.^2,2).^0.5,1)).^2)^0.5;
    %%%% group density
    D_temp=[];
    for j=1:size(xyz,1)
        Dist_rank=sum((xyz-xyz(j,:)).^2,2).^0.5;        
        D_temp=[D_temp;max(Dist_rank)];
    end
    D(i)=length(id_birds)/mean(D_temp)^3*6/pi;
end
N=length(id_birds);
save('data/mob_sample_midOrder.mat','t','p','D','N');
% assign one color for each trajectory
for j=1:length(id_birds)
    C{j}.color=rand(1,3); % color
end

figure('units','inches','position',[5 2 10 8]);
%%% create a projection plane on the horizontal plane
Pj_z=-20; 
X = [min(tracks_good(:,2))-1 max(tracks_good(:,2))+1 max(tracks_good(:,2))+1 min(tracks_good(:,2))-1];
Y = [min(tracks_good(:,3))-1 min(tracks_good(:,3))-1 max(tracks_good(:,3))+1 max(tracks_good(:,3))+1];
Z = [Pj_z-offset Pj_z-offset Pj_z-offset Pj_z-offset];
fill3(X,Y,Z,[0.8 0.8 0.8],'edgecolor','none');hold on 
plot3([X(1) X(1)+10],[Y(1) Y(1)],[Z(1) Z(1)],'k-','lineWidth',10); % scale bar
%%% plot 3D trajectories
for i=1:length(id_birds)
    id=find(tracks_good(:,1)==id_birds(i));
    xyz=tracks_good(id,2:4);
    u=tracks_good(id,6:8);
    plot3(xyz(:,1),xyz(:,2),xyz(:,3),'-','color',C{i}.color,'lineWidth',5);hold on 
    scatter3(xyz(end,1),xyz(end,2),xyz(end,3),300,[0.2 0.2 0.2],'o','filled');
    
    plot3(xyz(:,1),xyz(:,2),repmat(Pj_z,[size(xyz,1),1]),'-','color',[0.5 0.5 0.5],'lineWidth',3);hold on 
    scatter3(xyz(end,1),xyz(end,2),Pj_z,150,[0.2 0.2 0.2],'o','filled');
end
axis equal;grid off;box on;axis off
set(gca,'Position',[0.05 0.1 0.9 0.85])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on','boxstyle','full'); 
view([45 15]) 


%% figure 2c: mobbing flocks at high density
load('data/mob_01.mat');
id_birds=[38 39 40 41 44 45 47]; %% 106, good
tracks_good=selectBirds(tracks_filt,id_birds);
t=unique(tracks_good(:,5));
%%%%% calculate the time-dependent group order and density
for i=1:length(t)
    %%% select the birds at this time
    id=find(tracks_good(:,5)==t(i));
    xyz=tracks_good(id,2:4);
    u=tracks_good(id,6:8);
    %%%% group order
    p(i)=1/size(u,1)*sum((sum(u./sum(u.^2,2).^0.5,1)).^2)^0.5;
    %%%% group density
    D_temp=[];
    for j=1:size(xyz,1)
        Dist_rank=sum((xyz-xyz(j,:)).^2,2).^0.5;        
        D_temp=[D_temp;max(Dist_rank)];
    end
    D(i)=length(id_birds)/mean(D_temp)^3*6/pi;
end
N=length(id_birds);
save('data/mob_sample_highOrder.mat','t','p','D','N');
% assign one color for each trajectory
for j=1:length(id_birds)
    C{j}.color=rand(1,3); % color
end

figure('units','inches','position',[5 2 10 8]);
%%% create a projection plane on the horizontal plane
Pj_z=-20; 
X = [min(tracks_good(:,2))-1 max(tracks_good(:,2))+1 max(tracks_good(:,2))+1 min(tracks_good(:,2))-1];
Y = [min(tracks_good(:,3))-1 min(tracks_good(:,3))-1 max(tracks_good(:,3))+1 max(tracks_good(:,3))+1];
Z = [Pj_z-offset Pj_z-offset Pj_z-offset Pj_z-offset];
fill3(X,Y,Z,[0.8 0.8 0.8],'edgecolor','none');hold on 
plot3([X(2) X(2)-10],[Y(end) Y(end)],[Z(1) Z(1)],'k-','lineWidth',10); % scale bar
%%% plot 3D trajectories
for i=1:length(id_birds)
    id=find(tracks_good(:,1)==id_birds(i));
    xyz=tracks_good(id,2:4);
    u=tracks_good(id,6:8);
    plot3(xyz(:,1),xyz(:,2),xyz(:,3),'-','color',C{i}.color,'lineWidth',5);hold on 
%     scatter3(xyz(end,1),xyz(end,2),xyz(end,3),300,[0.2 0.2 0.2],'o','filled',...
%         'MarkerEdgeColor',C{i}.color,'lineWidth',3);
    scatter3(xyz(end,1),xyz(end,2),xyz(end,3),300,[0.2 0.2 0.2],'o','filled');
    
    plot3(xyz(:,1),xyz(:,2),repmat(Pj_z,[size(xyz,1),1]),'-','color',[0.5 0.5 0.5],'lineWidth',3);hold on 
    scatter3(xyz(end,1),xyz(end,2),Pj_z,150,[0.2 0.2 0.2],'o','filled');
end
axis equal;grid off;box on;axis off
set(gca,'Position',[0.05 0.1 0.9 0.85])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on','boxstyle','full'); 
view([225 15]) 

%% figure 2d, plot time-dependent group order for three examples
figure('units','inches','position',[5 2 3.2 2.2]);
load('data/mob_sample_lowOrder.mat');
h1=plot(t-min(t),p,'--','LineWidth',1.5,'color',[0.5 0.5 0.5]);hold on
load('data/mob_sample_midOrder.mat');
h2=plot(t-min(t),p,'-.','LineWidth',1.5,'color','b');hold on
load('data/mob_sample_highOrder.mat');
h3=plot(t-min(t),p,'-','LineWidth',1.5,'color','r');hold on
axis([0 10 0.1 1.1]);box off
set(gca,'Position',[0.15 0.15 0.75 0.75])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on','yscale','linear'); 
legend([h1 h2 h3],{'a','b','c'},...
    'FontSize',11,'FontWeight','Bold')