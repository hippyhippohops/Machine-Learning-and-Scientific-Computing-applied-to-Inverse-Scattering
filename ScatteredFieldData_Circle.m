%% Step Up: Introducing Variables and Fixing Parameters of Scatterer and Grid

% Wave Parameters
wave_number = 16; 
wave_frequency = 2 * pi / wave_number;
direction_of_incident_wave = [1;1]; 

% Scattering Object Parameters
radius_of_scatterer = 0.3;
center_of_scatterer = [-1;-1];  
permitivitty_of_scatterer = 0.3; 

%Setting up the grid for domain of Interest
x = linspace(-2, 2, 150);
y = linspace(-2, 2, 150);
size(x);
size(y);
[X, Y] = meshgrid(x, y);
size(X);
size(Y);

%% Calculating Incident Field in grid G

% the incident field = exp(ik x.d)
u_in = exp(1i*wave_number*(direction_of_incident_wave(1,1)*X + direction_of_incident_wave(2,1)*Y)); %Functions can ask component wise to 

%% Defining the scattering object and visualising it

% Defining the permitivitty function
ball = zeros(size(X));
ball((X-center_of_scatterer(1,1)).^2 + (Y-center_of_scatterer(2,1)).^2 <= (radius_of_scatterer)^2) = permitivitty_of_scatterer;

%Plotting the scattering object
figure;

%surf(X, Y, Z)
%colormap(1-gray); % Set the colormap to grayscale
surf(X, Y, ball); % Plot filled contours
colorbar; % Show colorbar
xlabel('x-axis');
ylabel('y-axis');
shading interp
title('Scattering Object');
view(2)

%% Calculating the scattered field data

[X_new, Y_new] = meshgrid(x+0.0001, y); % Shifted Grid to calculate scattered field

% Defining contract in shifted grid
contrast = zeros(size(X_new));
contrast((X_new - center_of_scatterer(1,1)).^2 + (Y_new - center_of_scatterer(2,1)).^2 <= (radius_of_scatterer)^2) = permitivitty_of_scatterer;

% Defining Incident wave in shifted grid
u_in_new_grid = exp(1i*wave_number*(direction_of_incident_wave(1,1)*X_new + direction_of_incident_wave(2,1)*Y_new));

scattered_field_data = zeros(size(X_new));

for k=1:length(x)
    for j=1:length(y)
        scattered_wave_at_point = 0;
        x_coordinate = x(k);
        y_coordinate = y(j);
        scattered_wave_at_point = sum(sum(u_in_new_grid .* contrast .* besselh(0,wave_number * sqrt((X_new - x_coordinate).^2 + (Y_new - y_coordinate).^2))));
        scattered_field_data(j,k) = scattered_wave_at_point;
        k+j;
    end
end

%% Visualising Real Scattered Field 

real_scattered_field_values = real(scattered_field_data);
figure;
surf(X, Y, real_scattered_field_values); % Plot filled contours
colorbar; % Show colorbar
xlabel('x-axis');
ylabel('y-axis');
shading interp
title('Real Scattered Field Data');
view(2)

%% Visualising Complex Scattered Field 

complex_scattered_field_values = imag(scattered_field_data);
figure;
surf(X, Y, complex_scattered_field_values); % Plot filled contours
colorbar; % Show colorbar
xlabel('x-axis');
ylabel('y-axis');
shading interp
title('Complex Scattered Field Data');
view(2)

