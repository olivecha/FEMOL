function [index]=elementdof(node,nnel,ndof)
%----------------------------------------------------------
%  Purpose:
%     Compute system dofs associated with each element 
%
%  Synopsis:
%     [index]=elementdof(node,nnel,ndof)
%
%  Variable Description:
%     index - system dof vector associated with element "iel"
%     iel - element number whose system dofs are to be determined
%     node - nodes associated with the element "iel"
%     nnel - number of nodes per element
%     ndof - number of dofs per node 
%-----------------------------------------------------------
 % HERE global dof are assigned in a such way that suppose if at node 1,
 % there are 3 displacement components, then Global dof are assigned as
 % 1,2,3 for same node 1.
 
   k=0;
   for i=1:nnel
     start = (node(i)-1)*ndof;
       for j=1:ndof
         k=k+1;
         index(k)=start+j;
       end
   end

 
