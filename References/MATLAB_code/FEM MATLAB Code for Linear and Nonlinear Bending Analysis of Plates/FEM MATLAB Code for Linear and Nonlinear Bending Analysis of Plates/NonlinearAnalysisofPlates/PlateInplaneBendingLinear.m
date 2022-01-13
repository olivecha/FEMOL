function [pb]=PlateInplaneBendingLinear(nnel,dshapedx,dshapedy)

%--------------------------------------------------------------------------
%  Purpose:
%     determine the kinematic matrix expression relating linear inplane-bending actions
%
%  Synopsis:
%     [pb]=PlateInplaneBendingLinear(nnel,dshapedx,dshapedy) 
%
%  Variable Description:
%     nnel - number of nodes per element
%     dshapedx - derivatives of shape functions with respect to x   
%     dshapedy - derivatives of shape functions with respect to y
%--------------------------------------------------------------------------


 for i=1:nnel
 i1=(i-1)*5+1;  
 i2=i1+1;
 i3=i2+1;
 i4=i3+1;
 i5=i4+1;
 
 pb(1,i1)=dshapedx(i);
 pb(2,i2)=dshapedy(i);
 pb(3,i1)=dshapedy(i);
 pb(3,i2)=dshapedx(i);
 pb(4,i4)=-dshapedx(i);
 pb(5,i5)=-dshapedy(i);
 pb(6,i4)=-dshapedy(i);
 pb(6,i5)=-dshapedx(i);

 end
