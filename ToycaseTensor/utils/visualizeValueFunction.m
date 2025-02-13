function V_reshaped = visualizeValueFunction(V,x1l,x1u,x2l,x2u,l)

xLimits = [x1l, x1u];  % X-axis limits
yLimits = [x2l, x2u];  % Y-axis limits

% Calculate the number of grid points
numGridPointsX = l(1);
numGridPointsY = l(2);

stepSize = (x1u-x1l)/l(1);  % Step size for the grid

% Generate a grid of x and y coordinates
x = linspace(xLimits(1), xLimits(2), numGridPointsX);
y = linspace(yLimits(1), yLimits(2), numGridPointsY);

% Create a 2D grid of coordinates
[X, Y] = meshgrid(x, y);

% Assuming the value function V is of size (numGridPointsX * numGridPointsY, 1)
% Ensure that the size of V matches the required size
expectedSize = numGridPointsX * numGridPointsY;

% Reshape V to match the 2D grid
V_reshaped = reshape(V, [numGridPointsY, numGridPointsX]);
end