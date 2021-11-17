function [stiffness,tangentstiffness,force,bcdof]=SDF(P,E,nu,t,coordinates,nodes,nel,nnel,ndof,sdof,edof,displacement,typeBC,loadstep,updateintforce)

u = displacement(1:5:sdof) ;
v = displacement(2:5:sdof) ;
w = displacement(3:5:sdof) ;
tithax = displacement(4:5:sdof) ;
tithay = displacement(5:5:sdof) ;

force=zeros(sdof,1);                     % system total force vecotr  
stiffness=zeros(sdof,sdof);              % system stiffness matrix
index=zeros(edof,1);                     % index vector
tangentstiffness=zeros(sdof,sdof);       % system tangent stiffness matrix  


% Order of Gauss Quadrature
%--------------------------------------------------------------------------
[pointb,weightb]=  GaussQuadrature('second');     % sampling points & weights for bending
[points,weights]=   GaussQuadrature('first');     % sampling points & weights for shear
shcof = 5/6;                                      % shear correction factor

% Material Properties in form of matrix
Dp= [E*t/(1-nu^2)  E*t*nu/(1-nu^2) 0 ; E*t*nu/(1-nu^2)  E*t/(1-nu^2)   0; 0  0  E*t/(2*(1+nu))]; % in-plane material matrix
Db= [E*t^3/(12*(1-nu^2)) E*t^3*nu/(12*(1-nu^2)) 0; E*t^3*nu/(12*(1-nu^2)) E*t^3/(12*(1-nu^2)) 0;  0 0 E*t^3/(24*(1+nu))]; % bending material matrix
Ds=[E*t*shcof/(2*(1+nu)) 0; 0 E*t*shcof/(2*(1+nu))]; % shear material matrix



nglb=size(pointb,1);                     % 2x2 Gauss-Legendre quadrature for bending 
ngls=size(points,1);                     % 1x1 Gauss-Legendre quadrature for shear

    
for iel=1:nel                        % loop for the total number of elements

for i=1:nnel                        % loop for total number of nodes for single element   

node(i)=nodes(iel,i);               % extract connected node for (iel)^th element
xx(i)=coordinates(node(i),1);       % extract x value of the node
yy(i)=coordinates(node(i),2);       % extract y value of the node
u_p(i)= u(node(i));                  % extract value of u from previous iterations
v_p(i)=v(node(i));                   % extract value of v from previous iterations
w_p(i)= w(node(i));                  % extract value of w from previous iterations
tithax_p(i)=tithax(node(i));          % extract value of thetha_x from previous iterations
tithay_p(i)=tithay(node(i));          % extract value of thetha_y from previous iterations

end

%--------------------------------------------------------------------------
% Initialization of matrices and vectors
%--------------------------------------------------------------------------
KPB = zeros(edof,edof);              
KS = zeros(edof,edof);               
KPL=zeros(edof,edof);
KPLS = zeros(edof,edof); 
ksigma = zeros(edof,edof);               
f = zeros(edof,1) ;                      
%--------------------------------------------------------------------------
%  numerical integration for initial stress matrix
%--------------------------------------------------------------------------


for int=1:nglb                        % nglb is sampling points =4 
xi=pointb(int,1);                      
wt=weightb(int,1);                    
eta=pointb(int,2);                    


[shape,dshapedxi,dshapedeta]=Shapefunctions(xi,eta);    
    % compute shape functions and derivatives at sampling point

[detjacobian,invjacobian]=Jacobian(nnel,dshapedxi,dshapedeta,xx,yy);  % compute Jacobian

[dshapehdx,dshapehdy]=ShapefunctionDerivatives(nnel,dshapedxi,dshapedeta,invjacobian);
                                     % derivatives w.r.t. physical coordinate
                              
S=InitialStressVector(dshapehdx,dshapehdy,u_p,v_p,w_p,E,t,nu);    
G=ShapeDerivativeForInitialStressMatrix(nnel,dshapehdx,dshapehdy,shape);
%--------------------------------------------------------------------------
%  compute initial stress stiffness element matrix
%--------------------------------------------------------------------------


ksigma=ksigma+G'*S*G*wt*detjacobian; % this is an initial stress stiffness matrix

end                      % end of numerical integration loop


%--------------------------------------------------------------------------
%  numerical integration for linear inplane-bending matrix, nonlinear inplane matrix and  force vector
%--------------------------------------------------------------------------

for int=1:nglb                        % nglb is sampling points =4 
xi=pointb(int,1);                     
wt=weightb(int,1);                   
eta=pointb(int,2);                    


[shape,dshapedxi,dshapedeta]=Shapefunctions(xi,eta);    
    % compute shape functions and derivatives at sampling point

[detjacobian,invjacobian]=Jacobian(nnel,dshapedxi,dshapedeta,xx,yy);  % compute Jacobian

[dshapehdx,dshapehdy]=ShapefunctionDerivatives(nnel,dshapedxi,dshapedeta,invjacobian);
                                     % derivatives w.r.t. physical coordinate
                              

Bpb=PlateInplaneBendingLinear(nnel,dshapehdx,dshapehdy);           % linear inplane-bending kinematic matrix
Bpl=PlateInplaneNonlinear(nnel,dshapehdx,dshapehdy,u_p,v_p,w_p);    % nonlinear inplane kinematic matrix


%--------------------------------------------------------------------------
%  compute linear/nonlinear inplane/bending element matrix
%--------------------------------------------------------------------------

KPB=KPB+Bpb'*blkdiag(Dp,Db)*Bpb*wt*detjacobian; % Linear inplane-bending stiffness matrix
KPLS=KPLS+(0.5*Bpb'*blkdiag(Dp,Db)*Bpl+Bpl'*blkdiag(Dp,Db)*Bpb+0.5*Bpl'*blkdiag(Dp,Db)*Bpl)*wt*detjacobian; % nonlinear part of stiffness matix
KPL=KPL+(Bpb'*blkdiag(Dp,Db)*Bpl+Bpl'*blkdiag(Dp,Db)*Bpb+Bpl'*blkdiag(Dp,Db)*Bpl)*wt*detjacobian; % nonlinear part of tangent stiffness matrix


% converting uniform pressure into equivalent nodal forces. For this integration we are using two point Guass quadrature.
[fe] = Force(nnel,shape,P) ;             % Force vector
f = f+fe*wt*detjacobian ;
end                      % end of numerical integration loop


%--------------------------------------------------------------------------
%  numerical integration for shear matrix
%--------------------------------------------------------------------------

for int=1:ngls           % ngls is sampling points =2 
xi=points(int,1);                  
wt=weights(int,1);               
eta=points(int,2);                  


[shape,dshapedxi,dshapedeta]=Shapefunctions(xi,eta);   
        % compute shape functions and derivatives at sampling point

[detjacobian,invjacobian]=Jacobian(nnel,dshapedxi,dshapedeta,xx,yy);  % compute Jacobian

[dshapehdx,dshapehdy]=ShapefunctionDerivatives(nnel,dshapedxi,dshapedeta,invjacobian); 
            % derivatives w.r.t. physical coordinate


Bs=PlateShearLinear(nnel,dshapehdx,dshapehdy,shape);   % linear kinematix matrix for shear

%--------------------------------------------------------------------------
%  compute shear element matrix
%--------------------------------------------------------------------------
                
KS=KS+Bs'*Ds*Bs*wt*detjacobian;                      % linear shear stiffness matrix

end                      % end of numerical integration loop

%--------------------------------------------------------------------------
%  compute element matrix
%--------------------------------------------------------------------------
ke = KPB+KS+KPLS;       % This is element stiffness matrix
kt = KPB+KS+ksigma+KPL; % This is element tangent stiffness matrix

index=elementdof(node,nnel,ndof);% extract system dofs associated with element  

[stiffness,tangentstiffness,force]=Assemble(stiffness,tangentstiffness,force,ke,kt,f,index);      
                           % assemble element stiffness, tangent stiffness and force matrices into global stiffness, tangentstiffness and force matrix/vectors
end


%--------------------------------------------------------------------------
% Boundary conditions
%--------------------------------------------------------------------------

bcdof = BoundaryCondition(typeBC,coordinates,loadstep,updateintforce) ;

[stiffness,tangentstiffness,force] = Constraints(stiffness,tangentstiffness,force,bcdof); 
% this is modified matrices and vectors based on  boundary conditions.
