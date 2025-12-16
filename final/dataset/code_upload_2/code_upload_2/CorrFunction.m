function [r,C,GroupSZ]=CorrFunction(XYZ,u)
% function to calculate correlation function C(r/L), L is the group size
% inputs: XYZ -- position for each particle  
%         u -- full velocity for each particle
% outputs: r -- distance between birds
%          C -- correlation strength as a function of r
%          GroupSZ -- the group size defined as mean distance to the furthest individuals
Cor_all=[];
Dist_all=[];
%%% for each bird, get the velocity correlation/distance to all others
Uave=mean(u,1);
u=u-mean(u,1); %% fluctuation velocity
u=u/mean(sum(u.^2,2).^0.5); %% follow 'Attanasi et al., 2014, PLOS one, Swarm'
for j=1:size(XYZ,1)
    xyz1=XYZ(j,:);
    u1=u(j,:);
    for k=j+1:size(XYZ,1)
        xyz2=XYZ(k,:);
        u2=u(k,:);
        Dist_temp=sum((xyz1-xyz2).^2,2).^0.5;  
        Cor_temp=sum(u1.*u2); %%% correlation of orientation
        Cor_all=[Cor_all;Cor_temp];
        Dist_all=[Dist_all;Dist_temp];
    end
end

%%% get correlation function C(r)
edges=linspace(0,max(Dist_all),25);
for i=1:length(edges)-2
    id=find(Dist_all>edges(i) & Dist_all<edges(i+2));
    C(i)=mean(Cor_all(id));
    r(i)=mean(Dist_all(id));
end
% C=C/C(1);
%%% get group size
for j=1:size(XYZ,1)
    xyz1=XYZ(j,:);
    Dist_temp=sum((xyz1-XYZ).^2,2).^0.5;
    [Dist_temp,I]=sort(Dist_temp);
    D_max(j)=Dist_temp(end);
end
% GroupSZ=mean(D_max);
GroupSZ=max(D_max);

end
