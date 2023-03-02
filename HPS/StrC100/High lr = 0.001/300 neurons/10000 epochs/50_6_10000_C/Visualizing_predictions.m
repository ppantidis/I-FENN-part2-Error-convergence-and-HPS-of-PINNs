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
geom_xlimit = [coordlimits(1,1)-2 coordlimits(2,1)+2];
geom_ylimit = [coordlimits(1,2)-2 coordlimits(2,2)+2];



%%%%%%%%%%%%%%%%% Load ground truth data %%%%%%%%%%%%%%%%
data_StrC100 = table2array(readtable('data_GP_StrC100_Nonlocal_gradient_Analytical_inc_164_init.csv'));
xcoord_vec = data_StrC100(:,1);
ycoord_vec = data_StrC100(:,2);
local_strain_vec = data_StrC100(:,11);
nonlocal_strain_vec_truth = data_StrC100(:,12);
clear data_StrC100

%%%%%%%%%%%%%%%%% Load data from Neural Network prediction %%%%%%%%%%%%%%%%
% Load and store the nonlocal strain prediction vector

J_Coarse_50_6_10000_M1 = load(strcat("Model11_TestE_StrC100_Predictions_50hidlay_6units_10000ep_inc_164_v1.txt"));
J_Coarse_50_6_10000_M2 = load(strcat("Model11_TestE_StrC100_Predictions_50hidlay_6units_10000ep_inc_164_v2.txt"));
J_Coarse_50_6_10000_M3 = load(strcat("Model11_TestE_StrC100_Predictions_50hidlay_6units_10000ep_inc_164_v3.txt"));
J_Coarse_50_6_10000_M4 = load(strcat("Model11_TestE_StrC100_Predictions_50hidlay_6units_10000ep_inc_164_v4.txt"));
J_Coarse_50_6_10000_M5 = load(strcat("Model11_TestE_StrC100_Predictions_50hidlay_6units_10000ep_inc_164_v5.txt"));
J_Coarse_50_6_10000_M6 = load(strcat("Model11_TestE_StrC100_Predictions_50hidlay_6units_10000ep_inc_164_v6.txt"));
J_Coarse_50_6_10000_M7 = load(strcat("Model11_TestE_StrC100_Predictions_50hidlay_6units_10000ep_inc_164_v7.txt"));
J_Coarse_50_6_10000_M8 = load(strcat("Model11_TestE_StrC100_Predictions_50hidlay_6units_10000ep_inc_164_v8.txt"));
J_Coarse_50_6_10000_M9 = load(strcat("Model11_TestE_StrC100_Predictions_50hidlay_6units_10000ep_inc_164_v9.txt"));

nonlocal_strain_vec_pred_M1 = J_Coarse_50_6_10000_M1 / 10;
nonlocal_strain_vec_pred_M2 = J_Coarse_50_6_10000_M2 / 10;
nonlocal_strain_vec_pred_M3 = J_Coarse_50_6_10000_M3 / 10;
nonlocal_strain_vec_pred_M4 = J_Coarse_50_6_10000_M4 / 10;
nonlocal_strain_vec_pred_M5 = J_Coarse_50_6_10000_M5 / 10;
nonlocal_strain_vec_pred_M6 = J_Coarse_50_6_10000_M6 / 10;
nonlocal_strain_vec_pred_M7 = J_Coarse_50_6_10000_M7 / 10;
nonlocal_strain_vec_pred_M8 = J_Coarse_50_6_10000_M8 / 10;
nonlocal_strain_vec_pred_M9 = J_Coarse_50_6_10000_M9 / 10;


RSE_M1 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M1).^2 ./ nonlocal_strain_vec_truth.^2;
RSE_M2 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M2).^2 ./ nonlocal_strain_vec_truth.^2;
RSE_M3 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M3).^2 ./ nonlocal_strain_vec_truth.^2;
RSE_M4 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M4).^2 ./ nonlocal_strain_vec_truth.^2;
RSE_M5 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M5).^2 ./ nonlocal_strain_vec_truth.^2;
RSE_M6 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M6).^2 ./ nonlocal_strain_vec_truth.^2;
RSE_M7 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M7).^2 ./ nonlocal_strain_vec_truth.^2;
RSE_M8 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M8).^2 ./ nonlocal_strain_vec_truth.^2;
RSE_M9 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M9).^2 ./ nonlocal_strain_vec_truth.^2;


caxismax = 7e-4;

%%%%%%%%%%%%%%%%%%%%%%%% PLOTTING PREDICTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize figure
f = figure("Position",[200 80 820 700]);
hold on;
t = tiledlayout(3,3,'TileSpacing','tight','Padding','compact');
title0 = title(t, '\bf{Non-local strain predictions (50x6 - 10000)}');
set(title0,'interpreter','latex','fontsize', 16,'Color','k');

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,nonlocal_strain_vec_pred_M1,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet);caxis([0 caxismax]); box on;  % colorbar; 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,nonlocal_strain_vec_pred_M2,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 caxismax]); box on;  % colorbar;  

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,nonlocal_strain_vec_pred_M3,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 caxismax]); box on;  % colorbar; 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,nonlocal_strain_vec_pred_M4,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 caxismax]); box on; % colorbar; 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,nonlocal_strain_vec_pred_M5,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 caxismax]); box on; % colorbar; 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,nonlocal_strain_vec_pred_M6,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 caxismax]); box on;  % colorbar; 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]);
scatter(xcoord_vec,ycoord_vec,7,nonlocal_strain_vec_pred_M7,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 caxismax]); box on; % colorbar; 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]);
scatter(xcoord_vec,ycoord_vec,7,nonlocal_strain_vec_pred_M8,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 caxismax]); box on;  % colorbar; 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,nonlocal_strain_vec_pred_M9,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 caxismax]); box on;  % colorbar; 

print(f,'-dpdf', 'All_nonlocalstrain_predictions.pdf')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOTTING RSE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize figure
f = figure("Position",[200 80 820 700]);
hold on;
t = tiledlayout(3,3,'TileSpacing','tight','Padding','compact');
title0 = title(t, '\bf{Non-local strain RSE (50x6 - 10000)}');
set(title0,'interpreter','latex','fontsize', 16,'Color','k');

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,RSE_M1,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 5e-3]); box on;  colormap(gca,viridis); 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,RSE_M2,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 5e-3]); box on;  colormap(gca,viridis); 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,RSE_M3,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 5e-3]); box on;  colormap(gca,viridis); 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,RSE_M4,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 5e-3]); box on; colormap(gca,viridis); 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,RSE_M5,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 5e-3]); box on; colormap(gca,viridis);  

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,RSE_M6,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 5e-3]); box on; colormap(gca,viridis); 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]);
scatter(xcoord_vec,ycoord_vec,7,RSE_M7,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 5e-3]); box on; colormap(gca,viridis); 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]);
scatter(xcoord_vec,ycoord_vec,7,RSE_M8,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 5e-3]); box on;  colormap(gca,viridis); 

% Plot nonlocal equivalent strain contour prediction
nexttile; hold on; set(gca,'xtick',[],'ytick',[]); 
scatter(xcoord_vec,ycoord_vec,7,RSE_M9,'filled');
xlim(geom_xlimit); ylim(geom_ylimit); colormap(gca,jet); caxis([0 5e-3]); box on;  colormap(gca,viridis); 


print(f,'-dpdf', 'All_nonlocalstrain_RSE.pdf')



% -------------------------------------------------------------------------
L2RSE_vec_M1 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M1).^2 ./ nonlocal_strain_vec_truth.^2;
L2RSE_vec_M1(isnan(L2RSE_vec_M1)) = 0;
L2RSE_vec_M1(isinf(L2RSE_vec_M1)) = 0;
L2RSE_M1 = round(sqrt(sum(L2RSE_vec_M1)),2)

L2RSE_vec_M2 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M2).^2 ./ nonlocal_strain_vec_truth.^2;
L2RSE_vec_M2(isnan(L2RSE_vec_M2)) = 0;
L2RSE_vec_M2(isinf(L2RSE_vec_M2)) = 0;
L2RSE_M2 = round(sqrt(sum(L2RSE_vec_M2)),2)

L2RSE_vec_M3 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M3).^2 ./ nonlocal_strain_vec_truth.^2;
L2RSE_vec_M3(isnan(L2RSE_vec_M3)) = 0;
L2RSE_vec_M3(isinf(L2RSE_vec_M3)) = 0;
L2RSE_M3 = round(sqrt(sum(L2RSE_vec_M3)),2)

L2RSE_vec_M4 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M4).^2 ./ nonlocal_strain_vec_truth.^2;
L2RSE_vec_M4(isnan(L2RSE_vec_M4)) = 0;
L2RSE_vec_M4(isinf(L2RSE_vec_M4)) = 0;
L2RSE_M4 = round(sqrt(sum(L2RSE_vec_M4)),2)

L2RSE_vec_M5 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M5).^2 ./ nonlocal_strain_vec_truth.^2;
L2RSE_vec_M5(isnan(L2RSE_vec_M5)) = 0;
L2RSE_vec_M5(isinf(L2RSE_vec_M5)) = 0;
L2RSE_M5 = round(sqrt(sum(L2RSE_vec_M5)),2)

L2RSE_vec_M6 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M6).^2 ./ nonlocal_strain_vec_truth.^2;
L2RSE_vec_M6(isnan(L2RSE_vec_M6)) = 0;
L2RSE_vec_M6(isinf(L2RSE_vec_M6)) = 0;
L2RSE_M6 = round(sqrt(sum(L2RSE_vec_M6)),2)

L2RSE_vec_M7 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M7).^2 ./ nonlocal_strain_vec_truth.^2;
L2RSE_vec_M7(isnan(L2RSE_vec_M7)) = 0;
L2RSE_vec_M7(isinf(L2RSE_vec_M7)) = 0;
L2RSE_M7 = round(sqrt(sum(L2RSE_vec_M7)),2)

L2RSE_vec_M8 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M8).^2 ./ nonlocal_strain_vec_truth.^2;
L2RSE_vec_M8(isnan(L2RSE_vec_M8)) = 0;
L2RSE_vec_M8(isinf(L2RSE_vec_M8)) = 0;
L2RSE_M8 = round(sqrt(sum(L2RSE_vec_M8)),2)

L2RSE_vec_M9 = (nonlocal_strain_vec_truth - nonlocal_strain_vec_pred_M9).^2 ./ nonlocal_strain_vec_truth.^2;
L2RSE_vec_M9(isnan(L2RSE_vec_M9)) = 0;
L2RSE_vec_M9(isinf(L2RSE_vec_M9)) = 0;
L2RSE_M9 = round(sqrt(sum(L2RSE_vec_M9)),2)





























