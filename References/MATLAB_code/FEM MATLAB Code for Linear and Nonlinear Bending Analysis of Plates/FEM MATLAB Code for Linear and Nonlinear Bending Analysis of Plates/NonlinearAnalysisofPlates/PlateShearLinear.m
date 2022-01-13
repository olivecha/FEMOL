function [ps]=PlateShearLinear(nnel,dshapedx,dshapedy,shape)

%------------------------------------------------------------------------
%  Purpose:
%     determine the kinematic matrix expression relating shear action
%  Synopsis:
%     [ps]=PlateShearLinear(nnel,dshapedx,dshapedy,shape) 
%
%  Variable Description:
%     nnel - number of nodes per element
%     dshapedx - derivatives of shape functions with respect to x   
%     dshapedy - derivatives of shape functions with respect to y
%     shape - shape function
%------------------------------------------------------------------------

 for i=1:nnel
 i1=(i-1)*5+1;  
 i2=i1+1;
 i3=i2+1;
 i4=i3+1;
 i5=i4+1;
 
 ps(1,i3)=dshapedx(i);
 ps(1,i4)=-shape(i);
 ps(2,i3)=dshapedy(i);
 ps(2,i5)=-shape(i);

 

 end

