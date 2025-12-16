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
% calculate the local interactions in mobbing flocks:
%       r: 1st neighbour position relative to focal birds, focal birds located at (0,0,0)
%     Ang: alignment angle between focal bird with neighbour position
% Here, we fix n, and vary r

%% start
clear all;clc
%%%% mobbing flock #01
load('data/mob_01.mat');result_file='data/mob_interaction_01_n5.mat';
getInteractionMob(tracks_filt,result_file)

%%%% mobbing flock #02
load('data/mob_02.mat');result_file='data/mob_interaction_02_n5.mat';
getInteractionMob(tracks_filt,result_file)

%%%% mobbing flock #03
load('data/mob_03.mat');result_file='data/mob_interaction_03_n5.mat';
getInteractionMob(tracks_filt,result_file)

%%%% mobbing flock #07
load('data/mob_07.mat');result_file='data/mob_interaction_04_n5.mat';
getInteractionMob(tracks_filt,result_file)

%%%% mobbing flock #08
load('data/mob_08.mat');result_file='data/mob_interaction_05_n5.mat';
getInteractionMob(tracks_filt,result_file)

%%%% mobbing flock #08
load('data/mob_09.mat');result_file='data/mob_interaction_06_n5.mat';
getInteractionMob(tracks_filt,result_file)

function getInteractionMob(tracks_filt,result_file)
% calculate the local interactions 
%       r: 1st neighbour position relative to focal birds, focal birds located at (0,0,0)
%     Ang: alignment angle between focal bird with neighbour position
% save results in result_file
T=unique(tracks_filt(:,5)); % Time
tracks_filt(:,2:4)=tracks_filt(:,2:4)-mean(tracks_filt(:,2:4));
r=[]; 
Ang=[]; 
for i=1:length(T) % start for every time steps
    id=find(tracks_filt(:,5)==T(i));
    xyz=tracks_filt(id,2:4);
    u=tracks_filt(id,6:8);
    a=tracks_filt(id,9:11);
        
    if size(xyz,1)<6
        continue;
    end
    
    %%%% at each frame, treat every bird as a focal one
    for j=1:size(xyz,1)
        xyz1=xyz(j,:);u1=u(j,:);U1=sum(u1.^2)^0.5;
        
%         % if the focal birds are located on the edge, then ignore that bird
%         if abs(xyz1(1))>10 || abs(xyz(2))>10 || abs(xyz(3))>8  
%             continue;
%         end

        Dist_rank=sum((xyz1-xyz).^2,2).^0.5;
        [Dist_rank,I]=sort(Dist_rank);
        for k=6 % vary k from 2 to 6 to obtain results at _n1, _n2, _n3, _n4, _n5
%             if Dist_rank(k)>15
%                 continue;
%             end
            xyz2=xyz(I(k),:);u2=u(I(k),:);
            r_temp=xyz2-xyz1;
            r_temp=Rotate2U(r_temp,u1);
            r=[r;r_temp];
            Ang_temp=acos(sum(u1.*u2)./sqrt(sum(u1.*u1))./sqrt(sum(u2.*u2)))*180/pi; 
            Ang=[Ang;Ang_temp];
        end
    end
end
r_all=r;

save(result_file,'r','Ang'); % save results

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