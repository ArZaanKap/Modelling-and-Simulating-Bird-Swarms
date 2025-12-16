function tracks_good=selectBirds(tracks_filt,id_birds)
% select those trajectories given by the 'id_birds' 
% Date 2019/03
% writen by: Hangjian Ling 
%            Stanford University
% email:     linghj@stanford.edu

%%%%% select the time-period
id=find(tracks_filt(:,1)==id_birds(1));
t=tracks_filt(id,5);
ts=min(t);
te=max(t);
for i=2:length(id_birds)
    id=find(tracks_filt(:,1)==id_birds(i));
    t_temp=tracks_filt(id,5);
    if ts<min(t_temp)
        ts=min(t_temp);
    end
    if te>max(t_temp)
        te=max(t_temp);
    end
end
id=find(t>=ts & t<=te);
t=t(id)';

%%%% collect the data of the select birds
n=1;
for i=1:length(id_birds)
    %%%% search for each bird, get its trajectory
    id=find(tracks_filt(:,1)==id_birds(i) & tracks_filt(:,5)>=ts & tracks_filt(:,5)<=te);
    tracks_good(n:n+length(id)-1,:)=tracks_filt(id,:);
    n=n+length(id);
end

end