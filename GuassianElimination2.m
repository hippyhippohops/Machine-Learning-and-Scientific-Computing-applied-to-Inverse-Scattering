function[A] = GuassianElimination2(A) 
n = height(A);
for j = 1:(n-1)
        for i= (j+1) : n
            mult = A(i,j)/A(j,j) ;
            for k= j:n+1
                A(i,k) = A(i,k) - mult*A(j,k) ;
                A;
            end
        end
end
for p = n:-1:1
    for r = p+1:n
        x(p) = A(p,r)/A(p,r-1);
    end
end
A;
end