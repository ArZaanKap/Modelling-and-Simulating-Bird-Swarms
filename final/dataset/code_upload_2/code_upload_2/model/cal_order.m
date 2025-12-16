function [p_all,DS_all]=cal_order(folderName,N)
%%%%% function to post process modeling results, e.g. calcualte the
%%%%% polarization and density
p_all=[]; %% polarization
DS_all=[]; %% density
fileName=dir([folderName,sprintf('Sample_N%d_*.mat',N)]);
for k=1:size(fileName,1)  
    load([folderName,fileName(k,:).name]);
    NumBird=unique(track(:,1));
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
        D(i)=mean(D_temp);
    end
    p_all=[p_all,p];
    DS_all=[DS_all,length(NumBird)./D.^3*6/pi];
end

end