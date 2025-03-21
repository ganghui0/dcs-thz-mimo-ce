function plot_GeometryAE(p)

%%
global_verticesTx = zeros(3,p.Qat*p.Qt);
for mt = 1:p.Mt
    for nt = 1:p.Nt
        for mat = 1:p.Mat
            for nat = 1:p.Nat
                % qar = (mar-1)* p.Nar+ nar;
                qt = (mt-1)* p.Nt+ nt;
                qat = (mat-1)* p.Nat+ nat;
                global_verticesTx(:,(qt-1)*p.Qt+qat) = get_AEPosition(p, mt, nt, mat, nat, 'T','G');
            end
        end
    end
end
global_verticesTx = global_verticesTx.';

global_verticesRx = zeros(3,p.Qar*p.Qr);
for mr = 1:p.Mr
    for nr = 1:p.Nr
        for mar = 1:p.Mar
            for nar = 1:p.Nar
                % qar = (mar-1)* p.Nar+ nar;
                qr = (mr-1)* p.Nr+ nr;
                qar = (mar-1)* p.Nar+ nar;
                global_verticesRx(:,(qr-1)*p.Qr+qar) = get_AEPosition(p, mr, nr, mar, nar, 'R','G');
                % global_verticesRx(:,qar) = get_AEPosition(p, mr, nr, mar, nar, 'R');
            end
        end
    end
end
global_verticesRx = global_verticesRx.';

%%
% Determine the coordinate with the least variance
[~, I] = min(var(global_verticesRx));
global_verticesRx1 = global_verticesRx; 
% Remove the coordinate with the least variance
global_verticesRx1(:, I) = [];

% Use the convhull function to compute the convex hull
kRx = convhull(global_verticesRx1(:,1), global_verticesRx1(:,2));

% Determine the coordinate with the least variance
[~, I] = min(var(global_verticesTx));
global_verticesTx1 = global_verticesTx; 
% Remove the coordinate with the least variance
global_verticesTx1(:, I) = [];

% Use the convhull function to compute the convex hull
kTx = convhull(global_verticesTx1(:,1), global_verticesTx1(:,2));
% kTx = convhull(global_verticesTx);
% kRx = convhull(global_verticesRx);
% Plotting
% Defining global coordinate
% global_coord = [0,0,0];
% % Plot the global axis
% h1 = quiver3(global_coord(1), global_coord(2), global_coord(3), 1e-2, 0, 0, 'r', 'LineWidth', 2); hold on;
% h2 = quiver3(global_coord(1), global_coord(2), global_coord(3), 0, 1e-2, 0, 'g', 'LineWidth', 2); hold on;
% h3 = quiver3(global_coord(1), global_coord(2), global_coord(3), 0, 0, 1e-2, 'b', 'LineWidth', 2); hold on;

% Plot the Tx AEs
h4 = scatter3(global_verticesTx(:,1), global_verticesTx(:,2), global_verticesTx(:,3), 'MarkerEdgeColor', 'k');hold on
% Plot a plane through the vertices
h5 = fill3(global_verticesTx(kTx,1), global_verticesTx(kTx,2), global_verticesTx(kTx,3), 'k', 'FaceAlpha', 0.5);hold on

% Plot the Rx AEs
h6 = scatter3(global_verticesRx(:,1), global_verticesRx(:,2), global_verticesRx(:,3), 'MarkerEdgeColor', 'r');hold on
% Plot a plane through the vertices 
h7 = fill3(global_verticesRx(kRx,1), global_verticesRx(kRx,2), global_verticesRx(kRx,3), 'r', 'FaceAlpha', 0.5);hold on

% Make a legend
legend([h4 h5 h6 h7], {'Tx AEs', 'Tx Plane','Rx AEs', 'Rx Plane'});

grid on;
% axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');

