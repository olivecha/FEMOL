function [pb]=ShapeDerivativeForInitialStressMatrix(nnel,dshapedx,dshapedy,shape)


 for i=1:nnel
 i1=(i-1)*5+1;  
 i2=i1+1;
 i3=i2+1;
 i4=i3+1;
 i5=i4+1;
 
 pb(1,i3)=dshapedx(i);
 pb(1,i4)=0;
 pb(2,i3)=dshapedy(i);
 pb(2,i5)=0;

  end

