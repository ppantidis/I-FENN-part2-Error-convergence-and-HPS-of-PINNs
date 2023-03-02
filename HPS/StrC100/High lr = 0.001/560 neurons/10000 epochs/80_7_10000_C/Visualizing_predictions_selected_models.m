clc
clear all
close all
format compact

load custom_colormaps.mat

set(0,'DefaultAxesFontSize', 15, 'DefaultAxesLineWidth', 1, ...
'DefaultLineLineWidth', 2, 'DefaultAxesFontName', 'Latin Modern Math', ...
'DefaultTextFontName', 'Latin Modern Math', ...
'DefaultTextFontSize', 15, ...
'defaultAxesTickLabelInterpreter', 'latex', ...
'defaultLegendInterpreter', 'latex', ...
'defaultTextInterpreter', 'latex', ...
'defaultColorbarTickLabelInterpreter', 'latex', ...
'defaultPolaraxesTickLabelInterpreter', 'latex', ...
'defaultTextarrowshapeInterpreter', 'latex', ...
'defaultTextboxshapeInterpreter', 'latex', ...
'DefaultLegendBox','on', 'DefaultLegendFontSize', 17, ...
'DefaultAxesBoxStyle', 'back', 'DefaultAxesBox', 'off')


% Read data from the input file 
model_name = "StrC100";
infile     = fopen(model_name + "_parameters.txt",'r');
[SolverID,TangentID,ncoord,ndof,lc, ... 
            increment,inc_success_counter,min_iter,max_iter,max_accept_iter, ... 
            loadfactor,dlfactor,dlfactor_incr_threshold,increment_plot_threshold,loadfactor_plot_threshold, ...
            flaglf,countflaglf,incrflag,flagplot, ...
            ndomains,nprops,materialprops,alpha_val,beta_val,e_delta,dmax, ...
            nnodes,coords,nelem,maxnodes,connect,nelnodes,elident_vec,nfix,fixnodes] = func_read_input_file(infile); 

fclose(infile);

coordlimits = [min(coords'); max(coords')];
geom_xlimit = [coordlimits(1,1)-5 coordlimits(2,1)+5];
geom_ylimit = [coordlimits(1,2)-5 coordlimits(2,2)+5];



%%%%%%%%%%%%%%%%% Load ground truth data %%%%%%%%%%%%%%%%
data_StrC100 = table2array(readtable('data_GP_StrC100_Nonlocal_gradient_Analytical_inc_164_init.csv'));
xcoord_vec = data_StrC100(:,1);
ycoord_vec = data_StrC100(:,2);
local_strain_vec = data_StrC100(:,11);
nonlocal_strain_vec_truth = data_StrC100(:,12);
clear data_StrC100

%%%%%%%%%%%%%%%%% Load data from Neural Network prediction %%%%%%%%%%%%%%%%
% Load and store the nonlocal strain prediction vector

J_Coarse_80_7_10000_M8       = load(strcat("Model11_TestE_StrC100_Predictions_80hidlay_7units_10000ep_inc_164_v8.txt"));
nonlocal_strain_vec_pred_M8 = J_Coarse_80_7_10000_M8 / 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOTTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize figure
f = figure("Position",[200 180 450 330]);
hold on;
%t = tiledlayout(1,1,'TileSpacing','Normal','Padding','compact');
title0 = title('\bf{Predictions Model 8}');
set(title0,'interpreter','latex','fontsize',14);

% Plot nonlocal equivalent strain contour prediction
set(gca,'xtick',[],'ytick',[]); box on; % set(gca,'xtick',[]); 
scatter(xcoord_vec,ycoord_vec,12,nonlocal_strain_vec_pred_M8,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); hcb = colorbar;
caxis([0 4e-4]);
% set(gca,'ColorScale','log')

% text(5,35,"\bf{TRIVIAL SOLUTION}",'color','r','fontsize',17,'rotation',15);
% newTick = 0.669763e-5;
% pos = hcb.Position;
% r = (newTick - min(hcb.Ticks))/(max(hcb.Ticks)-min(hcb.Ticks));
% a = annotation('line', 'Position', [0.8104 0.23 0.0474 0],'color', [1,0,0], 'linewidth', 2);

print(f,'-dpdf', 'Predictions_80x7_10000_highlr_M8.pdf');




