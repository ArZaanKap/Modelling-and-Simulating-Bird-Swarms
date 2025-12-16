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
% Figure 1a-b: show bird trajectories in 3D (colored) and projected on 2D
% plane (grey)

%% start to plot
clear all;clc
%% Figure 1a, plot for one of the transit flock
load('data/transit_01.mat');Frames=550; % load data and select which frame to plot
% shift coordinate such x1 is align with flight direction
T=unique(tracks_filt(:,5));
curFrame=T(Frames);
id=find(tracks_filt(:,5)==curFrame);
u=tracks_filt(id,6:8);
U=mean(u,1);   
theta_temp=atan(U(2)/U(1));
R = [cos(theta_temp) -sin(theta_temp); sin(theta_temp) cos(theta_temp)];
UR=U(1:2)*R;UR(3)=U(3);
for j=1:size(tracks_filt,1)
    if UR(1)>0
        tracks_filt(j,2:3) = tracks_filt(j,2:3)*R; 
        tracks_filt(j,6:7)=tracks_filt(j,6:7)*R;
    elseif UR(1)<0
        tracks_filt(j,2:3) = tracks_filt(j,2:3)*R*(-1); 
        tracks_filt(j,6:7)=tracks_filt(j,6:7)*R*(-1);  
    end
end
% assign one color for each trajectory
NumT=unique(tracks_filt(id,1));
for j=1:length(NumT)
    C{j}.color=rand(1,3); 
end

figure('units','inches','position',[5 2 10 8]);

%%% create a projection plane on the horizontal plane
Pj_z=-25; 
X = [-25 35 35 -25];
Y = [-28 -28 32 32];
Z = [Pj_z-1 Pj_z-1 Pj_z-1 Pj_z-1];
fill3(X,Y,Z,[0.8 0.8 0.8],'edgecolor','none');hold on 
plot3([X(1) X(1)+10],[Y(1) Y(1)],[Z(1) Z(1)],'k-','lineWidth',10); % scale bar

%%% plot 3D trajectories
for i=1:length(NumT)
    Id=find(tracks_filt(:,1)==NumT(i));% & tracks_filt(:,5)<=curFrame);
    xyz=tracks_filt(Id,2:4);
    plot3(xyz(:,1),xyz(:,2),xyz(:,3),'-','color',C{i}.color,'lineWidth',3);hold on 
    scatter3(xyz(end,1),xyz(end,2),xyz(end,3),65,[0.2 0.2 0.2],'o','filled');
    
    plot3(xyz(:,1),xyz(:,2),repmat(Pj_z,[size(xyz,1),1]),'-','color',[0.5 0.5 0.5],'lineWidth',2.5);hold on 
    scatter3(xyz(end,1),xyz(end,2),Pj_z,65,[0.2 0.2 0.2],'o','filled');
end
axis equal;grid off;box on;axis off
set(gca,'Position',[0.05 0.1 0.9 0.85])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on','boxstyle','full'); 
view([45 15])

%% Figure 1b, plot for one of the mobbing flock
load('data/mob_03.mat');Frames=4200; % load data and select which frame to plot
% shift coordinate such x1 is align with flight direction
T=unique(tracks_filt(:,5));
curFrame=T(Frames);
id=find(tracks_filt(:,5)==curFrame);
u=tracks_filt(id,6:8);
U=mean(u,1);   
theta_temp=atan(U(2)/U(1));
R = [cos(theta_temp) -sin(theta_temp); sin(theta_temp) cos(theta_temp)];
UR=U(1:2)*R;UR(3)=U(3);
for j=1:size(tracks_filt,1)
    if UR(1)>0
        tracks_filt(j,2:3) = tracks_filt(j,2:3)*R; 
        tracks_filt(j,6:7)=tracks_filt(j,6:7)*R;
    elseif UR(1)<0
        tracks_filt(j,2:3) = tracks_filt(j,2:3)*R*(-1); 
        tracks_filt(j,6:7)=tracks_filt(j,6:7)*R*(-1);  
    end
end
% assign one color for each trajectory
NumT=unique(tracks_filt(id,1));
for j=1:length(NumT)
    C{j}.color=rand(1,3); 
end

figure('units','inches','position',[5 2 10 8]);

%%% create a projection plane on the horizontal plane
Pj_z=-20; 
X = [-22 20 20 -22];
Y = [-19 -19 23 23];
Z = [Pj_z-1 Pj_z-1 Pj_z-1 Pj_z-1];
fill3(X,Y,Z,[0.8 0.8 0.8],'edgecolor','none');hold on %%% create a projection plane in xy
plot3([X(1) X(1)+10],[Y(1) Y(1)],[Z(1) Z(1)],'k-','lineWidth',10);

%%% plot 3D trajectories
for i=1:length(NumT)
    Id=find(tracks_filt(:,1)==NumT(i));% & tracks_filt(:,5)<=curFrame);
    xyz=tracks_filt(Id,2:4);
    plot3(xyz(:,1),xyz(:,2),xyz(:,3),'-','color',C{i}.color,'lineWidth',3);hold on 
    scatter3(xyz(end,1),xyz(end,2),xyz(end,3),65,[0.2 0.2 0.2],'o','filled');
    
    plot3(xyz(:,1),xyz(:,2),repmat(Pj_z,[size(xyz,1),1]),'-','color',[0.5 0.5 0.5],'lineWidth',2.5);hold on 
    scatter3(xyz(end,1),xyz(end,2),Pj_z,65,[0.2 0.2 0.2],'o','filled');
end
axis equal;grid off;box on;axis off
set(gca,'Position',[0.05 0.1 0.9 0.85])
set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
    'XMinorTick','on','YMinorTick','on','boxstyle','full'); 
view([45 15])