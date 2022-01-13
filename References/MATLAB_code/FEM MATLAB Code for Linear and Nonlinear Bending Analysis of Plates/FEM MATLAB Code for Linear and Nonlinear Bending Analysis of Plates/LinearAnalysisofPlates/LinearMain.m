% Problem :  Linear Bending Analysis of  Mindlin plate  
% Two Boundary conditions are used, simply supported and clamped
%--------------------------------------------------------------------------
% Code for linear analysis of plates is written by Siva Srinivas Kolukula, http://www.mathworks.com/matlabcentral/fileexchange/32029-plate-bending
% Code for linear analysis of plates can be found in A.J.M. Ferreira, MATLAB Codes for Finite Element Analysis, Springer Publications,2008.
%
% Code modified by authors : Amit Patil, E-mail :Amit Patil,aspatil07@gmail.com 
%                            Siva Srinivas Kolukula, Email:allwayzitzme@gmail.com
%                    
%--------------------------------------------------------------------------
%----------------------------------------------------------------------------
%
%   Variable descriptions                                                      
%   ke = element stiffness matrix for bending + shear                                            
%   kb = element stiffness matrix for bending
%   ks = element stiffness matrix for shear 
%   f = element force vector
%   stiffness = system stiffness matrix                                             
%   force = system force vector                                                 
%   displacement = system nodal displacement vector
%   coordinates = coordinate values of each node
%   nodes = nodal connectivity of each element
%   index =  vector containing system dofs associated with each element     
%   pointb = matrix containing sampling points for bending term
%   weightb = vector containing weighting coefficients for bending term
%   points = matrix containing sampling points for shear term
%   weights = vector containing weighting coefficients for shear term
%   bcdof =  vector containing dofs associated with boundary conditions                                                  
%   B_pb = kinematic matrix for bending
%   D_pb = material matrix for bending
%   B_ps = kinematic matrix for shear
%   D_ps = material matrix for shear
%
%%

%----------------------------------------------------------------------------            
clear 
clc
disp('Please wait Programme is under Run')
%--------------------------------------------------------------------------
% Transverse uniform pressure on plate
%--------------------------------------------------------------------------
P = -1 ; 

%--------------------------------------------------------------------------
% Geometrical and material properties of plate
%--------------------------------------------------------------------------
a = 1;                            % length of the plate (along X-axis)
b = 1 ;                           % breadth of the plate (along Y-axis)
E = 10920;                        % elastic modulus
nu = 0.3;                         % poisson's ratio
t = 0.1 ;                         % plate thickness

%Number of mesh element in x and y direction
Nx=10;
Ny=10;
%--------------------------------------------------------------------------
% Input data for nodal connectivity for each element
%--------------------------------------------------------------------------

[coordinates, nodes] = MeshRectanglularPlate(a,b,Nx,Ny,1) ; % for node connectivity counting starts from 1 towards +y axis and new row again start at y=0. see pdf for more details.

nel = length(nodes) ;                  % number of elements
nnel=4;                                % number of nodes per element
ndof=3;                                % number of dofs per node
nnode = length(coordinates) ;          % total number of nodes in system
sdof=nnode*ndof;                       % total system dofs  
edof=nnel*ndof;                        % degrees of freedom per element

%--------------------------------------------------------------------------
% Order of Gauss Quadrature
%--------------------------------------------------------------------------
nglb=4;                     % 2x2 Gauss-Legendre quadrature for bending 
ngls=1;                     % 1x1 Gauss-Legendre quadrature for shear 

%%
%--------------------------------------------------------------------------
% Initialization of matrices and vectors
%--------------------------------------------------------------------------
force = zeros(sdof,1) ;             % system Force Vector
stiffness=zeros(sdof,sdof);         % system stiffness matrix
index=zeros(edof,1);                % index vector


%--------------------------------------------------------------------------
%  Computation of element matrices and vectors and their assembly
%--------------------------------------------------------------------------
%
%  For bending stiffness
%
[pointb, weightb]=GaussQuadrature('second');     % sampling points & weights
D_pb= E/(1-nu*nu)*[1  nu 0; nu  1  0; 0  0  (1-nu)/2];  % material matrix for bending
%
%  For shear stiffness
%
[points,weights] = GaussQuadrature('first');    % sampling points & weights
G = 0.5*E/(1+nu);                               % shear modulus
shcof = 5/6;                                    % shear correction factor
D_ps=G*shcof*[1 0; 0 1];                        % material matrix for shear 


for iel=1:nel                        % loop for the total number of elements

    for i=1:nnel                        % loop for total number of nodes for single element   
    node(i)=nodes(iel,i);               % extract connected node for (iel)^th element
    xx(i)=coordinates(node(i),1);       % extract x value of the node
    yy(i)=coordinates(node(i),2);       % extract y value of the node
    end
    
B_pb=zeros(3,edof);                 % initialization of kinematic matrix for bending
B_ps=zeros(2,edof);                 % initialization of kinematic matrix for shear
ke = zeros(edof,edof);              % initialization of element stiffness matrix 
kb = zeros(edof,edof);              % initialization of bending stiffness matrix 
ks = zeros(edof,edof);              % initialization of shear stiffness matrix 
f = zeros(edof,1) ;                 % initialization of force vector

%--------------------------------------------------------------------------
%  Numerical integration for bending term
%--------------------------------------------------------------------------
    for int=1:nglb                        % nglb is sampling points for bending 
    xi=pointb(int,1);                     % sampling point in x-axis here value of xi is given
    eta=pointb(int,2);                    % sampling point in y-axis. here value of eta is given
    wt=weightb(int,1);                    % weights for sampling points

    [shape,dshapedxi,dshapedeta]=Shapefunctions(xi,eta);    
        % compute shape functions and derivatives at sampling point

    [detjacobian,invjacobian]=Jacobian(nnel,dshapedxi,dshapedeta,xx,yy);  % compute Jacobian

    [dshapedx,dshapedy]=ShapefunctionDerivatives(nnel,dshapedxi,dshapedeta,invjacobian);
                                         % derivatives w.r.t. physical coordinate


    % The DOF for single element are ordered as u=[w thetax thetay], where thetax is rotation about y axis, this is important to define before formulating B matrix.

    B_pb=PlateBending(nnel,dshapedx,dshapedy);    % bending kinematic matrix

    %--------------------------------------------------------------------------
    %  compute bending element matrix
    %--------------------------------------------------------------------------

    kb= kb + t^3/12 * B_pb' * D_pb * B_pb * wt * detjacobian;

    end                      % end of numerical integration loop for bending term

%%
%--------------------------------------------------------------------------
%  numerical integration for shear term
%--------------------------------------------------------------------------

for int=1:ngls
xi=points(int,1);                  % sampling point in x-axis
eta=points(int,2);                  % sampling point in y-axis
wt=weights(int,1);               % weights for sampling points

[shape,dshapedxi,dshapedeta]=Shapefunctions(xi,eta);   
        % compute shape functions and derivatives at sampling point

[detjacobian,invjacobian]=Jacobian(nnel,dshapedxi,dshapedeta,xx,yy);  % compute Jacobian

[dshapedx,dshapedy]=ShapefunctionDerivatives(nnel,dshapedxi,dshapedeta,invjacobian); 
            % derivatives w.r.t. physical coordinate


B_ps=PlateShear(nnel,dshapedx,dshapedy,shape);        % shear kinematic matrix

%--------------------------------------------------------------------------
%  compute shear element matrix
%--------------------------------------------------------------------------
                
ks= ks + t * B_ps' * D_ps * B_ps * wt * detjacobian;

% converting uniform pressure into equivalent nodal forces. For this
% integration we are using one point Guass quadrature.
fe = Force(nnel,shape,P) ;             % Force vector
f = f+fe*wt*detjacobian ;


end                      % end of numerical integration loop for shear term

%--------------------------------------------------------------------------
%  compute element matrix
%--------------------------------------------------------------------------

ke = kb+ks;

index=elementdof(node,nnel,ndof);% extract system dofs associated with element  

[stiffness,force]=assemble(stiffness,force,ke,f,index);      
                           % assemble element stiffness matrix and force vector 
end
%--------------------------------------------------------------------------
% Boundary conditions
%--------------------------------------------------------------------------
typeBC = 'ss-ss-ss-ss' ;        % Boundary Condition type simply supported or clamped
%typeBC = 'c-c-c-c'   ;
bcdof = BoundaryCondition(typeBC,coordinates,1) ;

assembled_stiffness = stiffness;
assembled_force = force;

[stiffness,force] = constraints(stiffness,force,bcdof);

constr_stiffness = stiffness;
constr_force = force;

%--------------------------------------------------------------------------
% Solution
%--------------------------------------------------------------------------
displacement = stiffness\force ;
%--------------------------------------------------------------------------
% Output of displacements
%--------------------------------------------------------------------------
w = displacement(1:3:sdof) ;
thetax= displacement(2:3:sdof) ;
thetay = displacement(3:3:sdof) ;
% Maximum transverse displacement
format long 
minw = min(w) 



