clc
clear all
close all
format compact

set(0,'DefaultAxesFontSize', 12, 'DefaultAxesLineWidth', 1, ...
'DefaultLineLineWidth', 1.5, 'DefaultAxesFontName', 'Latin Modern Math', ...
'DefaultTextFontName', 'Latin Modern Math', ...
'DefaultTextFontSize', 12, ...
'defaultAxesTickLabelInterpreter', 'latex', ...
'defaultLegendInterpreter', 'latex', ...
'defaultTextInterpreter', 'latex', ...
'defaultColorbarTickLabelInterpreter', 'latex', ...
'defaultPolaraxesTickLabelInterpreter', 'latex', ...
'defaultTextarrowshapeInterpreter', 'latex', ...
'defaultTextboxshapeInterpreter', 'latex', ...
'DefaultLegendBox','on', 'DefaultLegendFontSize', 14, ...
'DefaultAxesBoxStyle', 'back', 'DefaultAxesBox', 'off')

% Hyperparameters
layers  = [70];      % 8 12 20 32;
neurons = [6];      % 8 12 20 32;
epochs  = [10000];   
runs    = 1:10;
Predictions_LshapedCoarse_70_6_10000_highlr       = zeros(16400,10);
Adam_Predictions_LshapedCoarse_70_6_10000_highlr  = zeros(16400,10);

LshapedCoarse_l2norm_70_6_10000_highlr        = zeros(length(runs),1);
LshapedCoarse_Adam_l2norm_70_6_10000_highlr   = zeros(length(runs),1);

% -------------------------------------------------------------------------
LshapedCoarse_all_true_vals       = table2array(readtable("data_GP_LshapedCoarse_Nonlocal_gradient_Analytical_inc_100_init.csv"));
LshapedCoarse_enonlocal_true_vals = LshapedCoarse_all_true_vals(:,12);

% -------------------------------------------------------------------------
% Adam
for i = 1:length(layers)
    for j = 1:length(neurons)
        for k = 1:length(epochs)
            for l = 1:length(runs)

                % Load file
                current_file = load(['Model11_TestE_LshapedCoarse_Adam_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_100_v', num2str(runs(l)), '.txt']);
                Adam_Predictions_LshapedCoarse_70_6_10000_highlr(:,l) = current_file / 10;
                LshapedCoarse_Adam_l2norm_70_6_10000_highlr(l,1) = sqrt(sum((LshapedCoarse_enonlocal_true_vals - Adam_Predictions_LshapedCoarse_70_6_10000_highlr(:,l)).^2 ./ LshapedCoarse_enonlocal_true_vals.^2));
                clear current_file

                % Load file
                current_file = load(['Model11_TestE_LshapedCoarse_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_100_v', num2str(runs(l)), '.txt']);
                Predictions_LshapedCoarse_70_6_10000_highlr(:,l) = current_file / 10;
                LshapedCoarse_l2norm_70_6_10000_highlr(l,1) = sqrt(sum((LshapedCoarse_enonlocal_true_vals - Predictions_LshapedCoarse_70_6_10000_highlr(:,l)).^2 ./ LshapedCoarse_enonlocal_true_vals.^2));
                clear current_file

            end        
        end
    end
end

save Adam_Pred_L2norms_LshapedCoarse_70_6_10000_highlr LshapedCoarse_Adam_l2norm_70_6_10000_highlr
save LBFGS_Pred_L2norms_LshapedCoarse_70_6_10000_highlr LshapedCoarse_l2norm_70_6_10000_highlr




