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
% generate results for Figure 1c-d: 
% calculate the local interactions in transit flocks:
%       r: 1st neighbour position relative to focal birds, focal birds located at (0,0,0)
%     Ang: alignment angle between focal bird with neighbour position

%% start
clear all;clc
%%% transit flock #01
load('data/transit_01.mat');Frames=200:650;result_file='data/transit_interaction_01_n5.mat';
getInteraction_Transit(tracks_filt,Frames,result_file)

%%% transit flock #02
load('data/transit_02.mat');Frames=750:950;result_file='data/transit_interaction_02_n5.mat';
getInteraction_Transit(tracks_filt,Frames,result_file)

%%% transit flock #03
load('data/transit_03.mat');Frames=50:400;result_file='data/transit_interaction_03_n5.mat';
getInteraction_Transit(tracks_filt,Frames,result_file)

%%% transit flock #04
load('data/transit_04.mat');Frames=550:900;result_file='data/transit_interaction_04_n5.mat';
getInteraction_Transit(tracks_filt,Frames,result_file)

%%% transit flock #05
load('data/transit_05.mat');Frames=100:300;result_file='data/transit_interaction_05_n5.mat';
getInteraction_Transit(tracks_filt,Frames,result_file)

%%% transit flock #06
load('data/transit_06.mat');Frames=600:900;result_file='data/transit_interaction_06_n5.mat';
getInteraction_Transit(tracks_filt,Frames,result_file)

function getInteraction_Transit(tracks_filt,Frames,result_file)
% calculate the local interactions 
%       r: 1st neighbour position relative to focal birds, focal birds located at (0,0,0)
%     Ang: alignment angle between focal bird with neighbour position
% save results in result_file

T=unique(tracks_filt(:,5));
r=[]; 
Ang=[]; 
for i=Frames
    id=find(tracks_filt(:,5)==T(i));
    XYZ=tracks_filt(id,2:4);
    u_all=tracks_filt(id,6:8);
    NumT=unique(tracks_filt(id,1));
    if length(NumT)<50
        continue;
    end
    for j=1:length(NumT)
        Dist_rank=sum((XYZ(j,:)-XYZ).^2,2).^0.5;
        [Dist_rank,I]=sort(Dist_rank);        
        xyz1=XYZ(j,:);u1=u_all(j,:);
        xyz2=XYZ(I(6),:);u2=u_all(I(6),:); % change from I(2) to I(6) to obtain results at _n1, _n2, _n3, _n4, _n5
        %%%% treat bird 1 as focal bird
        r_temp=xyz2-xyz1; %% focal bird at (0,0)
        r_temp=Rotate2U(r_temp,u1);
        %%%% alignment angle difference
        Ang_temp=acos(sum(u1.*u2)./sqrt(sum(u1.*u1))./sqrt(sum(u2.*u2)))*180/pi; 
        
        r=[r;r_temp];
        Ang=[Ang;Ang_temp];
    end
end
save(result_file,'r','Ang');

% % % % %%%%% calcualte anistropic factor, and mean Distance
% % % % load('neighbor_1.mat')
% % % D=sum(r.^2,2).^0.5; %% total distance 
% % % D_xy=sum(r(:,1:2).^2,2).^0.5; %% distance in horizontal plane
% % % uij=r(:,1:2)./D_xy;
% % % mean(D(:))
% % % median(uij(:,2).*uij(:,2)-uij(:,1).*uij(:,1))

end


function XYZ=Rotate2U(XYZ,U)
%%%% move coordinate to along flight direction   
theta=atan(U(2)/U(1));
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
U(1:2)=U(1:2)*R;
for j=1:size(XYZ,1)
    XYZ(j,1:2) = XYZ(j,1:2)*R; %% R_xy(j,1) is along flight direction
end    
end