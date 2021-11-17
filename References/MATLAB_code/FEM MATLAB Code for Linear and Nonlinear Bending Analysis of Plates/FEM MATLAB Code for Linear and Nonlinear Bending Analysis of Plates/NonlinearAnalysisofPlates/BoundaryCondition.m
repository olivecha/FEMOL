function bcdof = BoundaryCondition(typeBC,coordinates,loadstep,updateintforce)

%--------------------------------------------------------------------------
%   Purpose:
%           To determine the boundary conditions degree of freedom
%   Synopsis:
%           bcdof = BoundaryCondition(typeBC,coordinates,loadstep,updateintforce)
%
%   Variable Description:
%          bcdof - boundary condition degree of freedom
%          typeBC - string which gives the state of boundary condition
%          coordinates - geometric coordinates of nodes
%           
%--------------------------------------------------------------------------

L1 = find(coordinates(:,2)==min(coordinates(:,2))) ; % at y = 0 (along X-axes)
L2 = find(coordinates(:,1)==max(coordinates(:,1))) ; % at x = a (along Y-axes)
L3 = find(coordinates(:,2)==max(coordinates(:,2))) ; % at y = b (along X-axes)
L4 = find(coordinates(:,1)==min(coordinates(:,1))) ; % at x = 0 (along Y-axes)
m = length(L1) ;
n = length(L2);
 
switch typeBC
          
        
    case 'ss-ss-ss-ss'
        if (loadstep==1 && updateintforce==1)
            disp('plate is simply supported at all the edges')
        end
        dofL1 = zeros(1,4*m) ;
        dofL2 = zeros(1,4*n) ;
        dofL3 = zeros(1,4*m) ;
        dofL4 = zeros(1,4*n) ;
        for i = 1:m   % moving along x axis
            i1 = 4*(i-1)+1 ;
            i2 = i1+1 ;
            i3 = i2+1 ;
            i4 =  i3+1;
            dofL1(i1) = 5*L1(i)-4;  % constraining u along x axis
            dofL1(i2) = 5*L1(i)-3 ; % constraining v along x axis
            dofL1(i3) = 5*L1(i)-2 ; % constraining w along x axis
            dofL1(i4) = 5*L1(i)-1 ; % constraining tithax along x axis
            dofL3(i1) = 5*L3(i)-4 ;
            dofL3(i2) = 5*L3(i)-3 ;
            dofL3(i3) = 5*L3(i)-2 ;
            dofL3(i4) = 5*L3(i)-1 ;
            
        end   
        for i = 1:n
            i1 = 4*(i-1)+1 ;
            i2 = i1+1 ;
            i3 = i2+1 ;
            i4 = i3+1;
            dofL2(i1) = 5*L2(i)-4 ;  % constraining u along y axis
            dofL2(i2) = 5*L2(i)-3 ;  % constraining v along y axis
            dofL2(i3) = 5*L2(i)-2 ;  % constraining w along y axis
            dofL2(i4) = 5*L2(i) ;    % constraining tithay along y axis
            dofL4(i1) = 5*L4(i)-4 ;
            dofL4(i2) = 5*L4(i)-3 ;
            dofL4(i3) = 5*L4(i)-2 ;  
            dofL4(i4) = 5*L4(i) ;  
        end
        L1UL3 = union(dofL1,dofL3) ;
        L2UL4 = union(dofL2,dofL4) ;
        bcdof = union(L1UL3,L2UL4) ;
        
    case 'c-c-c-c'
        if (loadstep==1 && updateintforce==1)
            disp('plate is clamped at all the edges')
        end
        dofL1 = zeros(1,4*m) ;
        dofL2 = zeros(1,4*n) ;
        dofL3 = zeros(1,4*m) ;
        dofL4 = zeros(1,4*n) ;
        for i = 1:m
            i1 = 4*(i-1)+1 ;
            i2 = i1+1 ;
            i3 = i2+1 ;
            i4 = i3+1 ;
            dofL1(i1) = 5*L1(i)-4;  % constraining u along x axis
            dofL1(i2) = 5*L1(i)-3;  % constraining v along x axis
            dofL1(i3) = 5*L1(i)-2 ; % constraining w along x axis
            dofL1(i4) = 5*L1(i) ;   % constraining tithay along x axis
            dofL3(i1) = 5*L3(i)-4 ;
            dofL3(i2) = 5*L3(i)-3 ;
            dofL3(i3) = 5*L3(i)-2 ; 
            dofL3(i4) = 5*L3(i); 
        end   
        for i = 1:n
            i1 = 4*(i-1)+1 ;
            i2 = i1+1 ;
            i3 = i2+1 ;
            i4 = i3+1 ;
            dofL2(i1) = 5*L2(i)-4 ;    % constraining u along y axis
            dofL2(i2) = 5*L2(i)-3 ;    % constraining v along y axis
            dofL2(i3) = 5*L2(i)-2 ;    % constraining w along y axis
            dofL2(i4) = 5*L2(i)-1 ;    % constraining tithax along x axis 
            dofL4(i1) = 5*L4(i)-4 ;
            dofL4(i2) = 5*L4(i)-3 ;
            dofL4(i3) = 5*L4(i)-2 ; 
            dofL4(i4) = 5*L4(i)-1 ; 
        end
        L1UL3 = union(dofL1,dofL3) ;
        L2UL4 = union(dofL2,dofL4) ;
        bcdof = union(L1UL3,L2UL4) ;
end