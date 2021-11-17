function [pb]=PlateInplaneNonlinear(nnel,dshapedx,dshapedy,u_p,v_p,w_p)

%--------------------------------------------------------------------------
%  Purpose:
%     determine the kinematic matrix expression relating nonlinear inplane action
%  Synopsis:
%     [pb]=PlateInplaneNonlinear(nnel,dshapedx,dshapedy,u_p,v_p,w_p) 
%
%  Variable Description:
%     nnel - number of nodes per element
%     dshapedx - derivatives of shape functions with respect to x   
%     dshapedy - derivatives of shape functions with respect to y
%--------------------------------------------------------------------------

dudx=dshapedx(1)*u_p(1)+dshapedx(2)*u_p(2)+dshapedx(3)*u_p(3)+dshapedx(4)*u_p(4);
dudy=dshapedy(1)*u_p(1)+dshapedy(2)*u_p(2)+dshapedy(3)*u_p(3)+dshapedy(4)*u_p(4);
dvdx=dshapedx(1)*v_p(1)+dshapedx(2)*v_p(2)+dshapedx(3)*v_p(3)+dshapedx(4)*v_p(4);
dvdy=dshapedy(1)*v_p(1)+dshapedy(2)*v_p(2)+dshapedy(3)*v_p(3)+dshapedy(4)*v_p(4);
dwdx=dshapedx(1)*w_p(1)+dshapedx(2)*w_p(2)+dshapedx(3)*w_p(3)+dshapedx(4)*w_p(4);
dwdy=dshapedy(1)*w_p(1)+dshapedy(2)*w_p(2)+dshapedy(3)*w_p(3)+dshapedy(4)*w_p(4);

 for i=1:nnel
 i1=(i-1)*5+1;  
 i2=i1+1;
 i3=i2+1;
 i4=i3+1;
 i5=i4+1;
 
 pb(1,i3)=dwdx*dshapedx(i);
 pb(2,i3)=dwdy*dshapedy(i);
 pb(3,i3)=dwdy*dshapedx(i)+dwdx*dshapedy(i);
 pb(4,i4)=0;
 pb(5,i5)=0;
 pb(6,i5)=0;

 end
