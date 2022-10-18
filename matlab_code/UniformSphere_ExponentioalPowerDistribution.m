%This function recieves the following parameteres (p and R) and generates N uniform points on the generalized sphere defined
%by Exponentioal Power Distribution method
    %N: The number of samples that we need to generate
    %p: This vector containing the exponents Pi, i=1,...,n of the desired generalized sphere
    %R: Super reduis for the desired generalized sphere
function u = UniformSphere_ExponentioalPowerDistribution(N,p,R)
m=length(p);
u=zeros(m,N);
for i=1:N
v=gg6(m,1,0,1,p);
T=sum(abs(v').^p);
u(:,i)=(R/T).^(1./p).*v';
end