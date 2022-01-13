function [dshapedx,dshapedy]=ShapefunctionDerivatives(nnel,dshapehdxi,dshapedeta,invjacob)

%------------------------------------------------------------------------
%  Purpose:
%     determine derivatives of  isoparametric Q4 shape functions with 
%     respect to physical coordinate system
%
%  Synopsis:
%     [dhdx,dhdy]=shapefunctionderivatives(nnel,dhdr,dhds,invjacob)  
%
%  Variable Description:
%     dhdx - derivative of shape function w.r.t. physical coordinate x
%     dhdy - derivative of shape function w.r.t. physical coordinate y
%     nnel - number of nodes per element   
%     dshapedxi - derivative of shape functions w.r.t. natural coordinate xi
%     dshapedeta - derivative of shape functions w.r.t. natural coordinate eta
%     invjacob - inverse of  Jacobian matrix
%------------------------------------------------------------------------
% dN/dxi=Inv(J)*dN/dxi
 for i=1:nnel
 dshapedx(i)=invjacob(1,1)*dshapehdxi(i)+invjacob(1,2)*dshapedeta(i);
 dshapedy(i)=invjacob(2,1)*dshapehdxi(i)+invjacob(2,2)*dshapedeta(i);
 end
