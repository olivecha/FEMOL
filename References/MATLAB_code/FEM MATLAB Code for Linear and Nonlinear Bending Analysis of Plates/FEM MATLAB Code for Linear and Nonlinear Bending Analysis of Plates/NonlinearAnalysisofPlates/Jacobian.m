function [detjacobian,invjacobian]=Jacobian(nnel,dshapedxi,dshapedeta,xcoord,ycoord)

%------------------------------------------------------------------------
%  Purpose:
%     determine the Jacobian for two-dimensional mapping
%
%  Synopsis:
%     [detjacobian,invjacobian]=Jacobian(nnel,dhdr,dhds,xcoord,ycoord) 
%
%  Variable Description:
%     jacobian - Jacobian for one-dimension
%     nnel - number of nodes per element   
%     dshapedxi - derivative of shape functions w.r.t. natural coordinate xi
%     dshapedeta - derivative of shape functions w.r.t. natural coordinate eta
%     xcoord - x axis coordinate values of nodes
%     ycoord - y axis coordinate values of nodes
%------------------------------------------------------------------------

 jacobian=zeros(2,2);
% Jacobian first element is given by dx/dxi= d(N_i*x_i)/dxi=
% dshapedxi(1)*x(1)+dshapedxi(2)*x(2)+dshapedxi(3)*x(3)+dshapedxi(4)*x(4)
 for i=1:nnel
 jacobian(1,1) = jacobian(1,1)+dshapedxi(i)*xcoord(i);
 jacobian(1,2) = jacobian(1,2)+dshapedxi(i)*ycoord(i);
 jacobian(2,1) = jacobian(2,1)+dshapedeta(i)*xcoord(i);
 jacobian(2,2) = jacobian(2,2)+dshapedeta(i)*ycoord(i);
 end
 detjacobian = det(jacobian) ;  % Determinant of Jacobian matrix
 invjacobian = inv(jacobian) ;  % Inverse of Jacobian matrix