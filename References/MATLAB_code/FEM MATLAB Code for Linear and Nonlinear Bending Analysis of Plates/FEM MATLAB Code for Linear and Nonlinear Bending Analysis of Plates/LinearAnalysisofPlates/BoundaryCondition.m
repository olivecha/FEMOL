function bcdof = BoundaryCondition(typeBC,coordinates,loadstep)

%--------------------------------------------------------------------------
%   Purpose:
%           To determine the boundary conditions degree of freedom
%   Synopsis:
%           bcdof = BoundaryCondition(typeBC,coordinates,loadstep)
%
%   Variable Description:
%           bcdof - boundary condition degree of freedom
%                   dof's are (w, thetax, thetay)
%           typeBC - string which gives the state of boundary condition
%           coordinates - geometric coordinates of nodes
%           
%--------------------------------------------------------------------------

L1 = find(coordinates(:,2)==min(coordinates(:,2))) ; % at y = 0 (along X-axes)
L2 = find(coordinates(:,1)==max(coordinates(:,1))) ; % at x = a (along Y-axes)
L3 = find(coordinates(:,2)==max(coordinates(:,2))) ; % at y = b (along X-axes)
L4 = find(coordinates(:,1)==min(coordinates(:,1))) ; % at x = 0 (along Y-axes)
m = length(L1) ;
n = length(L2) ;
 
switch typeBC
    case 'ss-ss-ss-ss'
        if loadstep==1
            disp('plate is simply supported at all the edges')
        end
        dofL1 = zeros(1,2*m) ;
        dofL2 = zeros(1,2*n) ;
        dofL3 = zeros(1,2*m) ;
        dofL4 = zeros(1,2*n) ;
        for i = 1:m
            i1 = 2*(i-1)+2 ;
            i2 = i1-1 ;
            dofL1(i1) = 3*L1(i)-1;    % constraining thetax along x axis
            dofL1(i2) = 3*L1(i)-2 ;   % constraining w along x axis
            dofL3(i1) = 3*L3(i)-1 ;    
            dofL3(i2) = 3*L3(i)-2 ;   
        end   
        for i = 1:n
            i1 = 2*(i-1)+2 ;
            i2 = i1-1 ;
            dofL2(i1) = 3*L2(i) ;    % constraining thetay along y axis
            dofL2(i2) = 3*L2(i)-2 ;   % constraining w along y axis
            dofL4(i1) = 3*L4(i) ;
            dofL4(i2) = 3*L4(i)-2 ;
        end
        L1UL3 = union(dofL1,dofL3) ;
        L2UL4 = union(dofL2,dofL4) ;
        bcdof = union(L1UL3,L2UL4) ;
        
    case 'c-c-c-c'
        if loadstep==1
        disp('plate is clamped at all the edges')
        end
        dofL1 = zeros(1,2*m) ;
        dofL2 = zeros(1,2*n) ;
        dofL3 = zeros(1,2*m) ;
        dofL4 = zeros(1,2*n) ;
        for i = 1:m
            i1 = 2*(i-1)+2 ;
            i2 = i1-1 ;
            dofL1(i1) = 3*L1(i);    % constraining thetay along x axis
            dofL1(i2) = 3*L1(i)-2 ; % constraining w along x axis
            dofL3(i1) = 3*L3(i) ;
            dofL3(i2) = 3*L3(i)-2 ;
        end   
        for i = 1:n
            i1 = 2*(i-1)+2 ;
            i2 = i1-1 ;
            dofL2(i1) = 3*L2(i)-1 ;  % constraining thetax along y axis
            dofL2(i2) = 3*L2(i)-2 ;  % constraining w along x axis
            dofL4(i1) = 3*L4(i)-1 ;
            dofL4(i2) = 3*L4(i)-2 ;
        end
        L1UL3 = union(dofL1,dofL3) ;
        L2UL4 = union(dofL2,dofL4) ;
        bcdof = union(L1UL3,L2UL4) ;
end