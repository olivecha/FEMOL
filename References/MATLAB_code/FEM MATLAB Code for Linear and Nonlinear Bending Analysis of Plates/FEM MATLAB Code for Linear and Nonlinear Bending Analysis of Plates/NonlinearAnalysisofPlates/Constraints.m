function [kss,ktt,FF]=Constraints(kss,ktt,FF,bcdof)

%----------------------------------------------------------
%  Purpose:
%     Apply constraints to matrix equation [K]{x}={F}
%
%  Synopsis:
%     [kss,ktt,FF]=Constraints(kss,ktt,FF,bcdof,bcval)
%
%  Variable Description:
%     kss - system stiffness matrix  
%     ktt  - system tangent stiffness matrix
%     ff - system force vector 
%     bcdof - a vector containging constrained d.o.f
%-----------------------------------------------------------
 
 n=length(bcdof);
 sdof=size(kss);
%
 for i=1:n
    c=bcdof(i);
    for j=1:sdof
     kss(c,j)=0;
     kss(j,c)=0;
     ktt(c,j)=0;
     ktt(j,c)=0;
    end

    kss(c,c)=1;
    ktt(c,c)=1;
    FF(c)=0;
end

