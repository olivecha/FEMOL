function [pb]=InitialStressVector(dshapedx,dshapedy,u_p,v_p,w_p,E,t,nu)

%--------------------------------------------------------------------------
%  Purpose:
%     determine the initial stress vector
%
%  Synopsis:
%    [pb]=InitialStressVector(dshapedx,dshapedy,u_p,v_p,w_p,E,t,nu)
%
%  Variable Description:
%     nnel - number of nodes per element
%     dshapedx - derivatives of shape functions with respect to x   
%     dshapedy - derivatives of shape functions with respect to y
%     u_p,v_p,w_p - displacement values
%     E,t,nu - Young modulus, thickness and Poissons ratio
%--------------------------------------------------------------------------

dudx=dshapedx(1)*u_p(1)+dshapedx(2)*u_p(2)+dshapedx(3)*u_p(3)+dshapedx(4)*u_p(4);
dudy=dshapedy(1)*u_p(1)+dshapedy(2)*u_p(2)+dshapedy(3)*u_p(3)+dshapedy(4)*u_p(4);
dvdx=dshapedx(1)*v_p(1)+dshapedx(2)*v_p(2)+dshapedx(3)*v_p(3)+dshapedx(4)*v_p(4);
dvdy=dshapedy(1)*v_p(1)+dshapedy(2)*v_p(2)+dshapedy(3)*v_p(3)+dshapedy(4)*v_p(4);
dwdx=dshapedx(1)*w_p(1)+dshapedx(2)*w_p(2)+dshapedx(3)*w_p(3)+dshapedx(4)*w_p(4);
dwdy=dshapedy(1)*w_p(1)+dshapedy(2)*w_p(2)+dshapedy(3)*w_p(3)+dshapedy(4)*w_p(4);


 
 pb(1,1)=E*t/(1-nu^2)*(dudx+0.5*(dwdx)^2)+E*t*nu/(1-nu^2)*(dvdy+0.5*(dwdy)^2);
 pb(1,2)=E*t*0.5/(1+nu)*(dudy+dvdx+dwdx*dwdy);
 pb(2,1)=E*t*0.5/(1+nu)*(dudy+dvdx+dwdx*dwdy);
 pb(2,2)=E*t*nu/(1-nu^2)*(dudx+0.5*(dwdx)^2)+E*t/(1-nu^2)*(dvdy+0.5*(dwdy)^2);
 


