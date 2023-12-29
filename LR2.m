function[L R]=LR2(A)

%Decomposition of Matrix AA: A = L R
z=size(A,1);
L=zeros(z,z);
R=zeros(z,z);

for i=1:z
    % Finding L
    for k=1:i-1
        L(i,k)=A(i,k);
        for j=1:k-1
            L(i,k)= L(i,k)-L(i,j)*R(j,k);
        end
        L(i,k) = L(i,k)/R(k,k);
    end
    
    % Finding R
    for k=i:z
        R(i,k) = A(i,k);
        for j=1:i-1
            R(i,k)= R(i,k)-L(i,j)*R(j,k);
        end
    end
end

R;
L;

end