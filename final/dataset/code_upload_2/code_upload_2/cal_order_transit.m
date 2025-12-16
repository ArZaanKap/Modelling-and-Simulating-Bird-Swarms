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
% generate results for Figure 3 and Extend Data Fig 2: 
%      calculate group density and order for transit flocks
%      at each focal bird, select a local neighbours including a total 5,
%      10 and 20 birds, and calculate its order and density

%% start
clear all;clc
LocalRange=20; % select subgroup size 5, 10 or 20
%%% transit flock #01
load('data/transit_01.mat');Frames=200:650;result_file='data/transit_order_01.mat';
getGroupOrder_Tran(tracks_filt,Frames,result_file,LocalRange)

%%% transit flock #02
load('data/transit_02.mat');Frames=750:950;result_file='data/transit_order_02.mat';
getGroupOrder_Tran(tracks_filt,Frames,result_file,LocalRange)

%%% transit flock #03
load('data/transit_03.mat');Frames=50:400;result_file='data/transit_order_03.mat';
getGroupOrder_Tran(tracks_filt,Frames,result_file,LocalRange)

%%% transit flock #04
load('data/transit_04.mat');Frames=550:900;result_file='data/transit_order_04.mat';
getGroupOrder_Tran(tracks_filt,Frames,result_file,LocalRange)

%%% transit flock #05
load('data/transit_05.mat');Frames=100:300;result_file='data/transit_order_05.mat';
getGroupOrder_Tran(tracks_filt,Frames,result_file,LocalRange)

%%% transit flock #06
load('data/transit_06.mat');Frames=600:900;result_file='data/transit_order_06.mat';
getGroupOrder_Tran(tracks_filt,Frames,result_file,LocalRange)

function getGroupOrder_Tran(tracks_filt,Frames,result_file,LocalRange)
%  at each focal bird, select a local neighbours including a total 5 birds, 
%  and calculate its order and density
%  save results in result_file
T=unique(tracks_filt(:,5));
order=[]; % subgroup polarization
density=[]; % subgroup density
for i=Frames
    id=find(tracks_filt(:,5)==T(i));
    xyz=tracks_filt(id,2:4);
    u=tracks_filt(id,6:8);
    NumT=unique(tracks_filt(id,1));
    if size(xyz,1)<LocalRange
        continue;
    end
    for j=1:size(xyz,1)
        Dist_rank=sum((xyz(j,:)-xyz).^2,2).^0.5;
        [Dist_rank,I]=sort(Dist_rank);
        %%%% select the local neighbor with 'LocalRange' birds
        BirdIds=NumT(I(1:LocalRange));
        xyz_l=xyz(I(1:LocalRange),:);
        u_l=u(I(1:LocalRange),:);
        %%%%% local order
        order=[order;1/LocalRange*sum((sum(u_l./sum(u_l.^2,2).^0.5,1)).^2)^0.5]; 
        
        %%%% local density
        D_temp=[];
        for j=1:size(xyz_l,1)
            Dist_rank=sum((xyz_l-xyz_l(j,:)).^2,2).^0.5;        
            D_temp=[D_temp;max(Dist_rank)];
        end
        density=[density;LocalRange/mean(D_temp)^3*6/pi];
    end
end
save(result_file,'density','order');

end