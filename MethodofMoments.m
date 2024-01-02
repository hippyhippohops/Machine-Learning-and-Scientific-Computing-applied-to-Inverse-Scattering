% The EM Wave Scattering gives us information about the electric field and
% magnetic field at each sampling point x_k in the Domain of Interest, D.
% Here, we will use these electric field and magnetic field at each
% sampling point x_k, to calculate the scattered field data at each x_k.

% Firstly, I would want access to the variables in the EM Wave Scattering
% script.So I would like to run that script:
run("EMWaveScattering.m");
result.ElectricField;
%% Gathering the necessary data we want

% result.ElectricField.x is a 11540 x 12 matrix. Why is only the last 2
% columns being displayed??
result.ElectricField.x;
real(result.ElectricField.x);
imag(result.ElectricField.x);
size(result.ElectricField.x); % 11540 x 12
size(real(result.ElectricField.x)); % 11540 x 12
size(imag(result.ElectricField.x)); % 11540 x 12

result.ElectricField.y;
real(result.ElectricField.y);
imag(result.ElectricField.y);
size(result.ElectricField.y); % 11540 x 12
size(real(result.ElectricField.y)); % 11540 x 12
size(imag(result.ElectricField.y)); % 11540 x 12

result.MagneticField.x;
% The above result doesn't return anything. So, for now, I will just assume
% it is all 0. but FIGURE OUT WHY
MagneticField_x = [];
MagneticField_x = zeros(11540,12);
real(MagneticField_x);
imag(MagneticField_x);
size(MagneticField_x); % 11540 x 12
size(real(MagneticField_x)); % 11540 x 12
size(imag(MagneticField_x)); % 11540 x 12

MagneticField_y = [];
MagneticField_y= zeros(11540,12);
real(MagneticField_y);
imag(MagneticField_y);
size(MagneticField_y); % 11540 x 12
size(real(MagneticField_y)); % 11540 x 12
size(imag(MagneticField_y)); % 11540 x 12

%% Compute M_k

%% Compute L_k

%% Compute N_k

%% First stage when k=m

% Computing A_kk

% Computing B_kk

% Computing C_kk

% Computing D_kk


%% First stage when k!=m

% Computing A_km

% Computing B_km

% Computing C_km

% Computing D_km

%% Forming the system of linear equation involving E(x_m) and H(x_m)
% and then solving it

%% Computing the electric and magnetic volume current density 
% J(x) -> electric volume current density 
% K(x) -> magnetic volume current density

%% Computing the scattered field data
