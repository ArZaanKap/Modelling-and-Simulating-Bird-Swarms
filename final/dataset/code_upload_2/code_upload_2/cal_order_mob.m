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
% generate results for Figure 3: 
%      calculate group density and order for a given ID numbers of birds
%      The ID numbers for 154 groups are list in 'Data/Mobbing_flocks.xlsx'
%      The 154 groups are selected from 10 mobbing flock events
%      The group selection criteria are: 
%        (i) the group size was larger than 3; 
%        (ii) the time duration was longer than 1.5 s; 
%        (iii) the jackdaws did not leave the measurement volume during the selected time period; 
%        (iv) there were no transitions from disordered to ordered states 
%                or from ordered to disordered states caused by, e.g. group fusion or fission

%% start
clear all;clc
load('data/mob_03.mat'); % select the mobbing flock event 
id_birds=[14 15 16 19 20 26 33 17]; % select the bird IDs (see 'Mobbing_flocks.xlsx') 
tracks_good=selectBirds(tracks_filt,id_birds);
t=unique(tracks_good(:,5));
%%%%% calculate the time-dependent variables
for i=1:length(t)
    %%% select the birds at this time
    id=find(tracks_good(:,5)==t(i));
    xyz=tracks_good(id,2:4);
    u=tracks_good(id,6:8);
    %%%% group order
    p(i)=1/size(u,1)*sum((sum(u./sum(u.^2,2).^0.5,1)).^2)^0.5;
    %%%% group density
    D_temp=[];
    D_temp2=[];
    for j=1:size(xyz,1)
        Dist_rank=sum((xyz-xyz(j,:)).^2,2).^0.5;   
        Dist_rank=sort(Dist_rank);
        D_temp=[D_temp;max(Dist_rank)];
        D_temp2=[D_temp2;Dist_rank(2)];
    end 
    D(i)=length(id_birds)/mean(D_temp)^3*6/pi; % based on largest distance
    
    D_local(i)=1/mean(D_temp2)^3*6/pi; % based on nearest distance
end
%%% mean group order 
prop.order=mean(p);
%%% group order standard deviation
prop.orderSD=std(p);
%%% mean group density 
prop.density=mean(D(i))*1000;
%%% group density standard deviation
prop.densitySD=std(D(i))*1000;
%%% mean group density 
prop.densitylocal=mean(D_local(i))*1000;
%%% time duration
prop.Time=max(t)-min(t);
prop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% extra plots %%%%%%%%%%%%%%%%%%%%%%%
% %%%% figure, birds trajectories
% for j=1:length(id_birds)
%     C{j}.color=rand(1,3); % color
% end
% figure('units','inches','position',[8 6 3.5 3]);
% for i=1:length(id_birds)
%     id=find(tracks_good(:,1)==id_birds(i));
%     xyz=tracks_good(id,2:4);
%     plot(xyz(:,1),xyz(:,2),'-','color',C{i}.color,'linewidth',1.5);hold on
% end
% id=find(tracks_good(:,5)==t(1));
% xyz=tracks_good(id,2:4);
% u=tracks_good(id,6:8);
% scatter(xyz(:,1),xyz(:,2),60,[0.5 0.5 0.5],'o','filled');
% quiver(xyz(:,1),xyz(:,2),u(:,1)/2,u(:,2)/2,'color',[0.5 0.5 0.5],...
%     'MaxHeadSize',1,'AutoScale','off','LineWidth',0.5);
% 
% id=find(tracks_good(:,5)==t(end));
% xyz=tracks_good(id,2:4);
% u=tracks_good(id,6:8);
% scatter(xyz(:,1),xyz(:,2),60,'k','o','filled');
% quiver(xyz(:,1),xyz(:,2),u(:,1)/2,u(:,2)/2,'r',...
%     'MaxHeadSize',1,'AutoScale','off','LineWidth',0.5);
% 
% axis equal;grid off;box on;
% set(gca,'Position',[0.13 0.13 0.8 0.8])
% set(gca,'FontSize',12,'TickLength',[0.03, 0.01],...
%     'XMinorTick','on','YMinorTick','on'); 
% 
% %%%% figure, time-dependent group order and density
% figure('units','inches','position',[5 2 3.5 3]);
% yyaxis left
% h1=plot(t-min(t),p,'-','LineWidth',1.5);hold on
% set(gca,'yscale','log','yTick',[0.1 0.2 0.5 1.0],'ylim',[0.1 1.1],'yscale','log'); 
% yyaxis right
% h2=plot(t-min(t),length(id_birds)./D.^3*6/pi,'--','LineWidth',1.5);hold on
% set(gca,'yscale','log','yTick',[0.001 0.01 0.1],'yscale','log'); 
% axis tight
% set(gca,'Position',[0.13 0.13 0.72 0.8])
% set(gca,'FontSize',12,'TickLength',[0.03, 0.01]); 
% legend([h1 h2],{'\phi','\rho'},...
%     'FontSize',14)
