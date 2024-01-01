%createpde("electromagnetic",ElectromagneticAnalysisType) returns an
%electromagnetic analysis model for the specified analysis type. In this
%case it is of harmonic analysis type.
emagmodel = createpde("electromagnetic","harmonic"); 

% Omega denotes the wave frequency
omega = 4*pi;

%square and diamond are 2 10 x 1 column matrices
square = [3; 4; -5; -5; 5; 5; -5; 5; 5; -5];
diamond = [2; 4; 2.1; 2.4; 2.7; 2.4; 1.5; 1.8; 1.5; 1.2];

% gd adjoins the 2 10 x 1 column matrices together, forming a 10 x 2 matrix
% gd is the geometry description matrix
gd = [square,diamond];

%sf is the set formula - FIGURE OUT WHAT THIS DOES!!!
% ns is the name space matrix: It is a text matrix that relates the columns
% in gd to variable names in sf
ns = char('square','diamond')'; 
sf = 'square - diamond';

%dl = decsg(gd,sf,ns) decomposes the geometry description matrix gd into
%the geometry matrix dl and returns the minimal regions that satisfy the
%set formula sf. The name-space matrix ns is a text matrix that relates the
%columns in gd to variable names in sf.
g = decsg(gd,sf,ns);
geometryFromEdges(emagmodel,g);


figure; 
%pdegplot(g) plots the geometry of a PDE problem, as described in g.
pdegplot(emagmodel,"EdgeLabels","on"); 
% xlim(limits) sets the x-axis limits for the current axes or char
xlim([-6,6]);
% ylim(limits) sets the x-axis limits for the current axes or char
ylim([-6,6]);

%VACUUM PROPERTIES
% In our PDE model, we set the vacuum permittivity and permeability to be 1.
%Vacuum permittivity is a physical constant that measures how well a vacuum
%allows the transmission of electric fieldlines.
%Vacuum Permeability is the magnetic permeability in a classical vacuum
emagmodel.VacuumPermittivity = 1;
emagmodel.VacuumPermeability = 1;

% Specify the relative permittivity, relative permeability, and
% conductivity of the material: This command assigns the relative
% permittivity, relative permeability, and conductivity to the entire
% geometry. Specify the permittivity and permeability of vacuum using the
% electromagnetic model properties. The solver requires all three
% parameters for a harmonic analysis.
electromagneticProperties(emagmodel,"RelativePermittivity",1,"RelativePermeability",1, "Conductivity",0);

%Apply the absorbing boundary condition on the edges of the square. Specify
%the thickness and attenuation rate for the absorbing region by using the
%Thickness, Exponent, and Scaling arguments.
electromagneticBC(emagmodel,"Edge",[1 2 7 8], "FarField","absorbing", "Thickness",2, "Exponent",4,"Scaling",1);

%Apply the boundary condition on the diamond edges.
innerBCFunc = @(location,~) [-exp(-1i*omega*location.x);
    zeros(1,length(location.x))];
bInner = electromagneticBC(emagmodel,"Edge",[3 4 5 6], "ElectricField",innerBCFunc);

%Generate a mesh
generateMesh(emagmodel,"Hmax",0.1);

%Solve the harmonic analysis model for the frequency ω=4π.
result = solve(emagmodel,"Frequency",omega);

%Plot the real part of the x-component of the resulting electric field.
u = result.ElectricField; 
% u is 11540 x 1 vector. what values do these contain

figure;
pdeplot(emagmodel,"XYData",real(u.Ex),"Mesh","off");
colormap(jet);

%Interpolate the resulting electric field to a grid covering the portion of
%the geometry, for x and y from -1 to 4.
v = linspace(-1,4,101);
[X,Y] = meshgrid(v);
Eintrp = interpolateHarmonicField(result,X,Y);

%Reshape Eintrp.Ex and plot the x-component of the resulting electric
%field.
EintrpX = reshape(Eintrp.ElectricField.Ex,size(X));

figure;
surf(X,Y,real(EintrpX),"LineStyle","none");
view(0,90);
colormap(jet);

%Using the solutions for a vector of frequencies, create an animation
%showing the corresponding solution to the time-dependent wave equation.
result = solve(emagmodel,"Frequency",omega/10:omega);
figure
for m = 1:length(omega/10:omega)
    u = result.ElectricField;
    pdeplot(emagmodel,"XYData",real(u.Ex(:,m)),"Mesh","off");
    colormap(jet)
    M(m) = getframe;
end
movie(M,5,3);



