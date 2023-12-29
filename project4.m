% Introduction of the project

% The goal of this project is as follow:
% (i) Finding an application of solving system of linear equations directly
%     - Finite Difference Method
% (ii) Studying the speed of the common methods we use here
%      - Guassian Elimination
%      - Lower Triangular Method
%      - Cholesky Factorisation
% (iii) Trying to Speed up this operation 
%      - Strassen Algorithm
%      - Winograd Convolutions

%% Introducing Finite Difference Methods

% The finite difference approximation for derivatives are one of the
% simplest and of the oldest methods to solve differential equations. WAs
% know by Euler in 1768 in 1d of space and known by Runge circa 1908

%The key idea involves approximating the differential operator by replacing
%the derivatives in the equation using difference quotients. The domain is
%partitioned in space and in time and approximations are computed at the
%space or time points. 

% The error between the numerical approximation and the exact solution is
% determined by the error that is commited by going from a differential
% operator to a difference operator. This error is called the
% discretization error or truncation error. This error stems from the fact
% that a finite part of the taylor series is used. 

%Explain first approximation 
%   - Use taylor series
%   - Use forward difference approximant
%   - Use back difference approximant 
%   - Combine forward and back difference approximant to get central
%   approximant for better accuracy 

% Explain 2nd approximation 
%   - Same principles as above

% Finite difference formulation for the One-dimensional problem


%% Solving the Problem by standard inversion 
% Here, we will discretise the non-homogenous Dirichlet Problem to find an
% approximation of the solution

% Define c in L^infinity(0,1) here
% c = e^(-x*2)

% Define f in L^2(0,1) here
% f(x) = x^2

discretization_intervals = linspace(10,100000,10);

number_of_points_interval_is_discretised_into = [];
time_taken = [];

for i=1:length(discretization_intervals) %does matlab have in function
    number_of_intervals=discretization_intervals(i);
    discretization_of_x_axis = linspace(0,1,number_of_intervals);

    %defining A_h
    A_h0 = spdiags([-1,2,1], -1:1, number_of_intervals);
    discretization_of_c = [];
    for j=1:length(discretization_of_x_axis)
        discretization_of_c = [discretization_of_c exp(-discretization_of_x_axis(j)^2)];
    end
    A_h =  1/(1/discretization_intervals(i)^2)*A_h0 + spdiags(discretization_of_c, 0, number_of_intervals);

    %Defining b_h
    discretization_of_f = [];
    for j=1:length(discretization_of_x_axis)
        discretization_of_f = [discretization_of_f discretization_of_x_axis(j)^2];
    end
    %Here, alpha=1
    discretization_of_f(1) = discretization_of_f(1)+ 1/(1/discretization_intervals(i)^2);
    %Here, beta=2
    discretization_of_f(end) = discretization_of_f(end)+ 2/(1/discretization_intervals(i)^2);
    discretization_of_x_axis;
    discretization_of_f;
    b_h = discretization_of_f;

    %finding u_h
    % variable for accumulating total time

    number_of_iterations = 100;
    averaging_time_taken = [];
    
    for j=1:number_of_iterations
        j;

        % do several tests, accumulating the total time
        tic;
        %finding u_h
        u_h = inv(A_h)*b_h;
        time_taken_for_computation = toc;
        if number_of_intervals == 100000
            plot(discretization_of_x_axis, u_h);
        end
    
        averaging_time_taken = [averaging_time_taken time_taken_for_computation];

    end

    averaging_time_taken = mean(averaging_time_taken);

    number_of_points_interval_is_discretised_into = [number_of_points_interval_is_discretised_into discretization_intervals(i)];
    time_taken = [time_taken averaging_time_taken];    
end 

figure;
plot(number_of_points_interval_is_discretised_into, time_taken);

time_taken_1 = time_taken;

% output the results
fprintf( '|     n | avg. time |\n' );
fprintf( '+-------+-----------+\n' );
for k=1:number_of_points_interval_is_discretised_into
    fprintf( '| %5d | %9.3e |\n', k, time_taken(k) );
end
fprintf('\n\n');

%% Applying the above to Guassian Elimination

% Here, we will discretise the non-homogenous Dirichlet Problem to find an
% approximation of the solution

% Define c in L^infinity(0,1) here
% c = e^(-x*2)

% Define f in L^2(0,1) here
% f(x) = x^2

discretization_intervals = linspace(10,100000,10);

number_of_points_interval_is_discretised_into = [];
time_taken = [];

for i=1:length(discretization_intervals) %does matlab have in function
    number_of_intervals=discretization_intervals(i);
    number_of_intervals
    discretization_of_x_axis = linspace(0,1,number_of_intervals);
    length(discretization_of_x_axis)

    %defining A_h
    A_h0 = spdiags([-1,2,1], -1:1, number_of_intervals);
    discretization_of_c = [];
    for j=1:length(discretization_of_x_axis)
        discretization_of_c = [discretization_of_c exp(-discretization_of_x_axis(j)^2)];
    end
    A_h =  1/(1/discretization_intervals(i)^2)*A_h0 + spdiags(discretization_of_c, 0, number_of_intervals);

    %Defining b_h
    discretization_of_f = [];
    for j=1:length(discretization_of_x_axis)
        discretization_of_f = [discretization_of_f discretization_of_x_axis(j)^2];
    end
    %Here, alpha=1
    discretization_of_f(1) = discretization_of_f(1)+ 1/(1/discretization_intervals(i)^2);
    %Here, beta=2
    discretization_of_f(end) = discretization_of_f(end)+ 2/(1/discretization_intervals(i)^2);
    discretization_of_x_axis;
    discretization_of_f;
    b_h = discretization_of_f;

    %finding u_h
    % variable for accumulating total time

    number_of_iterations = 100;
    averaging_time_taken = [];
    
    for j=1:number_of_iterations
        j;

        % do several tests, accumulating the total time
        tic;
        %finding u_h
        A_h = GuassianElimination2(A_h); % I am just factorising here, how do i actually solve for u_h
        u_h = inv(A_h)*b_h;
        time_taken_for_computation = toc;
    
        averaging_time_taken = [averaging_time_taken time_taken_for_computation];

    end

    if number_of_intervals == 100000
            plot(discretization_of_x_axis, u_h);
    end

    averaging_time_taken = mean(averaging_time_taken);

    number_of_points_interval_is_discretised_into = [number_of_points_interval_is_discretised_into discretization_intervals(i)];
    time_taken = [time_taken averaging_time_taken];    
end 

figure;
plot(number_of_points_interval_is_discretised_into, time_taken);

x = number_of_points_interval_is_discretised_into;
y = (2/3)*x.^3;

figure;
hold on;
plot(number_of_points_interval_is_discretised_into, time_taken, "b");
hold on;
plot(x,y, "k");

time_taken_2 = time_taken;

% output the results
fprintf( '|     n | avg. time |\n' );
fprintf( '+-------+-----------+\n' );
for k=1:number_of_points_interval_is_discretised_into
    fprintf( '| %5d | %9.3e |\n', k, time_taken(k) );
end
fprintf('\n\n'); 

%% Applying the above to Lower Upper Triangulation

% Here, we will discretise the non-homogenous Dirichlet Problem to find an
% approximation of the solution

% Define c in L^infinity(0,1) here
% c = e^(-x*2)

% Define f in L^2(0,1) here
% f(x) = x^2

discretization_intervals = linspace(10,100000,10);

number_of_points_interval_is_discretised_into = [];
time_taken = [];

for i=1:length(discretization_intervals) %does matlab have in function
    number_of_intervals=discretization_intervals(i);
    discretization_of_x_axis = linspace(0,1,number_of_intervals);

    %defining A_h
    A_h0 = spdiags([-1,2,1], -1:1, number_of_intervals);
    discretization_of_c = [];
    for j=1:length(discretization_of_x_axis)
        discretization_of_c = [discretization_of_c exp(-discretization_of_x_axis(j)^2)];
    end
    A_h =  1/(1/discretization_intervals(i)^2)*A_h0 + spdiags(discretization_of_c, 0, number_of_intervals);

    %Defining b_h
    discretization_of_f = [];
    for j=1:length(discretization_of_x_axis)
        discretization_of_f = [discretization_of_f discretization_of_x_axis(j)^2];
    end
    %Here, alpha=1
    discretization_of_f(1) = discretization_of_f(1)+ 1/(1/discretization_intervals(i)^2);
    %Here, beta=2
    discretization_of_f(end) = discretization_of_f(end)+ 2/(1/discretization_intervals(i)^2);
    discretization_of_x_axis;
    discretization_of_f;
    b_h = discretization_of_f;

    %finding u_h
    % variable for accumulating total time

    number_of_iterations = 100;
    averaging_time_taken = [];
    
    for j=1:number_of_iterations
        j;

        % do several tests, accumulating the total time
        tic;
        %finding u_h
        [L,R] = LR2(A_h); % I am just factorising here, how do i actually solve for u_h
        u_h = R'*(inv(L)*b_h);
        time_taken_for_computation = toc;
    
        averaging_time_taken = [averaging_time_taken time_taken_for_computation];

    end

    averaging_time_taken = mean(averaging_time_taken);

    number_of_points_interval_is_discretised_into = [number_of_points_interval_is_discretised_into discretization_intervals(i)];
    time_taken = [time_taken averaging_time_taken];    
end 

figure;
plot(number_of_points_interval_is_discretised_into, time_taken);

x = number_of_points_interval_is_discretised_into;
y = (2/3)*x.^3;

figure;
hold on;
plot(number_of_points_interval_is_discretised_into, time_taken, "b");
hold on;
plot(x,y, "k");

time_taken_3 = time_taken;

% output the results
fprintf( '|     n | avg. time |\n' );
fprintf( '+-------+-----------+\n' );
for k=1:number_of_points_interval_is_discretised_into
    fprintf( '| %5d | %9.3e |\n', k, time_taken(k) );
end
fprintf('\n\n'); 

%% Applying the above to Cholesky Factorisation

% Here, we will discretise the non-homogenous Dirichlet Problem to find an
% approximation of the solution

% Define c in L^infinity(0,1) here
% c = e^(-x*2)

% Define f in L^2(0,1) here
% f(x) = x^2

discretization_intervals = linspace(10,100000,10);

number_of_points_interval_is_discretised_into = [];
time_taken = [];

for i=1:length(discretization_intervals) %does matlab have in function
    number_of_intervals=discretization_intervals(i);
    discretization_of_x_axis = linspace(0,1,number_of_intervals);

    %defining A_h
    A_h0 = spdiags([-1,2,1], -1:1, number_of_intervals);
    discretization_of_c = [];
    for j=1:length(discretization_of_x_axis)
        discretization_of_c = [discretization_of_c exp(-discretization_of_x_axis(j)^2)];
    end
    A_h =  1/(1/discretization_intervals(i)^2)*A_h0 + spdiags(discretization_of_c, 0, number_of_intervals);

    %Defining b_h
    discretization_of_f = [];
    for j=1:length(discretization_of_x_axis)
        discretization_of_f = [discretization_of_f discretization_of_x_axis(j)^2];
    end
    %Here, alpha=1
    discretization_of_f(1) = discretization_of_f(1)+ 1/(1/discretization_intervals(i)^2);
    %Here, beta=2
    discretization_of_f(end) = discretization_of_f(end)+ 2/(1/discretization_intervals(i)^2);
    discretization_of_x_axis;
    discretization_of_f;
    b_h = discretization_of_f;

    %finding u_h
    % variable for accumulating total time

    number_of_iterations = 100;
    averaging_time_taken = [];
    
    for j=1:number_of_iterations
        j;

        % do several tests, accumulating the total time
        tic;
        %finding u_h
        A_h = Cholesky(A_h); % I am just factorising here, how do i actually solve for u_h
        u_h = A_h*A_h'*b_h;
        time_taken_for_computation = toc;
    
        averaging_time_taken = [averaging_time_taken time_taken_for_computation];

    end

    averaging_time_taken = mean(averaging_time_taken);

    number_of_points_interval_is_discretised_into = [number_of_points_interval_is_discretised_into discretization_intervals(i)];
    time_taken = [time_taken averaging_time_taken];    
end 

figure;
plot(number_of_points_interval_is_discretised_into, time_taken);

x = number_of_points_interval_is_discretised_into;
y = x.^3;

figure;
hold on;
plot(number_of_points_interval_is_discretised_into, time_taken, "b");
hold on;
plot(x,y, "k");

time_taken_4 = time_taken;

% output the results
fprintf( '|     n | avg. time |\n' );
fprintf( '+-------+-----------+\n' );
for k=1:number_of_points_interval_is_discretised_into
    fprintf( '| %5d | %9.3e |\n', k, time_taken(k) );
end
fprintf('\n\n'); 

%% Overall Comparison

figure;
hold on;
% Matlab Inversion
plot(number_of_points_interval_is_discretised_into, time_taken_1, "r");
hold on;
% Guass
plot(number_of_points_interval_is_discretised_into, time_taken_2, "g");
hold on;
% LU
plot(number_of_points_interval_is_discretised_into, time_taken_3, "b");
hold on;
%Cholesky
plot(number_of_points_interval_is_discretised_into, time_taken_4, "k");

