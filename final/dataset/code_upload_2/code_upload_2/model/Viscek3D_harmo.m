function Viscek3D_harmo(EventNum,N,noise)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vicsek 3D model with harmonic forces
% Manuscript: "Plasticity in local interactions and order transitions in flocking birds"
% Authors: Hangjian Ling et al.
% Manuscript submited to "Nature"
% Date 2019/03
% writen by: Hangjian Ling 
%            Stanford University
% email:     linghj@stanford.edu
% Reference: 1. Vicsek et al., 1995, Physical Review Letters
%            2. Attanasi et al., 2014, PLOS computational biology
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:length(EventNum)
    R=50; %% volume radius 
    u0=10; %% flight speed
    IR=4; %% interaction range
    Dt=0.05; %% time step ~ reaction time << IR/u0
    Rp=0.5; %% repulsion zone
    Potent=0.04; %% potential towards center
    %%%%% position, uniform within the sphere
    theta = asin(2*rand(N,1)-1); %% elevation angle, uniform distribution in (-pi/2,pi/2)
    phi = 2*pi*rand(N,1); %% azimuth angle, uniform distribution in (0, 2pi)
    radii = 0.8*R*(rand(N,1).^(1/3)); %% radius, not unifomr in (0, 0.8R)
    [x(:,1),x(:,2),x(:,3)] = sph2cart(phi,theta,radii); %% spherical to Cartesian coordinate
    %%%%% velocity, uniform within 4pi solid angle
    theta = asin(2*rand(N,1)-1); %% elevation angle, uniform distribution in (-pi/2,pi/2)
    phi = 2*pi*rand(N,1); %% azimuth angle, uniform distribution in (0, 2pi)
    [u(:,1),u(:,2),u(:,3)] = sph2cart(phi,theta,u0); %% spherical to Cartesian coordinate

%     %%%%%% show in a viedo
%     v = VideoWriter('test.avi');
%     v.FrameRate = 10;
%     v.Quality = 90;
%     open(v);

    track=[];
    Step=1;
    while Step<2*10^6/N %%% total step ~1/N such that data coverage
        %%%% update the velocity of each particle
        for i=1:N
            %%%% find neighbor for current particle
            x_temp=x(i,:); %% current position
            D=sum((x-x_temp).^2,2); 
            [D,I]=sort(D);
            id=find(D<IR);
            Neigh=I(id); %% metric; %%% Neigh=I(1:4); topo.

            %%%%%%%%%% now determine new velocity
            %%%% determine if exist neighbors in repulsion zone
            id=find(D<Rp & D>0);
            if ~isempty(id) %% exist particle in repulsion zone
                rij=x_temp-x(I(id),:);
                %%%% normalize rij to unit vector
                rij=rij./(sum(rij.^2,2).^0.5);
                u_ave=mean(rij,1);
            elseif isempty(id) %% no particle in repulsion zone
                %%%% get mean velocity of all neighbors
                u_ave=mean(u(Neigh,:),1);
                u_center=-1*Potent*x_temp;
                u_ave=u_ave+u_center;            
            end
            %%%% change u_ave to unit vector
            u_norm=u_ave/sum(u_ave.^2)^0.5;
            %%%%% add noise by rotating u_norm with a random angle
            [phi,theta,r_temp] = cart2sph(u_norm(:,1),u_norm(:,2),u_norm(:,3));
            phi=phi+(2*rand(1,1)-1)*pi*noise;
            theta=theta+asin(2*rand(1,1)-1)*noise;
            [u_rot(:,1),u_rot(:,2),u_rot(:,3)]=sph2cart(phi,theta,r_temp);
            u_new(i,:)=u_rot; 
        end
        %%% update
        u=u0*u_new;
        x=x+u*Dt;
        Step=Step+1;
        if Step>10000 && rem(Step,100)==0
            track_temp(:,1)=(1:N)';
            track_temp(:,5)=repmat(1,[N,1])*Step*Dt;
            track_temp(:,2:4)=x;
            track_temp(:,6:8)=u;
            track=[track;track_temp];
%             %%% plot results
%             cla;
%             set(gcf,'Color','w');
%             scatter3(x(:,1),x(:,2),x(:,3),'k.');hold on
%             quiver3(x(:,1),x(:,2),x(:,3),u(:,1),u(:,2),u(:,3),'r');
%             axis([-R R -R R -R R]);box on
%             set(gca,'Position',[0.15 0.15 0.78 0.78])
%             set(gca,'FontSize',20,'TickLength',[0.03, 0.01],...
%                 'XMinorTick','on','YMinorTick','on');        
%             frame = getframe(gcf);
%             writeVideo(v,frame);
        end
    end
%     close(v);
    save(['Results/',sprintf('Sample_N%d_%d.mat',N,EventNum(k))],'track')

end

end