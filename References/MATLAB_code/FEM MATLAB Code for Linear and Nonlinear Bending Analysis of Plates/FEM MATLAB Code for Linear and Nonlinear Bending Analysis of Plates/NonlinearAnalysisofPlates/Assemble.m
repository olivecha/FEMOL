function [kss,ktt,FF]=Assemble(kss,ktt,FF,ke,kt,f,index);      

%----------------------------------------------------------
%  Purpose:
%     Assembly of element stiffness matrix into the system matrix &
%     Assembly of element force vector into the system vector
%
%  Synopsis:
%     [kss,ktt,FF]=Assemble(kss,ktt,FF,ke,kt,f,index)
%
%  Variable Description:
%     kss - system stiffnes matrix
%     ktt - system tangentstiffness matrix
%     FF - system force vector
%     ke - element stiffness matrix
%     kt - element tangent stiffness matrix
%     f  - element force vector
%     index - d.o.f. vector associated with an element
%-----------------------------------------------------------


 edof = length(index);
 for i=1:edof
   ii=index(i);
     FF(ii)=FF(ii)+f(i);
     for j=1:edof
       jj=index(j);
         kss(ii,jj)=kss(ii,jj)+ke(i,j);
         ktt(ii,jj)=ktt(ii,jj)+kt(i,j);
     end
 end

