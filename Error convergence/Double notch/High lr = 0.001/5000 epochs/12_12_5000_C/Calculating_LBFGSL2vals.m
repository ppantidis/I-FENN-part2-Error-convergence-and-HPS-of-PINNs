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
layers  = [12];      % 8 12 20 32;
neurons = [12];      % 8 12 20 32;
epochs  = [5000];   
runs    = 1:10;
Predictions_DoubleNotchCoarse_12_12_5000_highlr       = zeros(7552,10);
Adam_Predictions_DoubleNotchCoarse_12_12_5000_highlr  = zeros(7552,10);

DoubleNotchCoarse_l2norm_12_12_5000_highlr        = zeros(length(runs),1);
DoubleNotchCoarse_Adam_l2norm_12_12_5000_highlr   = zeros(length(runs),1);

% -------------------------------------------------------------------------
DoubleNotchCoarse_all_true_vals       = table2array(readtable("data_GP_DoubleNotchCoarse_Nonlocal_gradient_Analytical_inc_140_init.csv"));
DoubleNotchCoarse_enonlocal_true_vals = DoubleNotchCoarse_all_true_vals(:,12);

% -------------------------------------------------------------------------
% Adam
for i = 1:length(layers)
    for j = 1:length(neurons)
        for k = 1:length(epochs)
            for l = 1:length(runs)

                % Load file
                current_file = load(['Model11_TestE_DoubleNotchCoarse_Adam_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_140_v', num2str(runs(l)), '.txt']);
                Adam_Predictions_DoubleNotchCoarse_12_12_5000_highlr(:,l) = current_file / 10;
                DoubleNotchCoarse_Adam_l2norm_12_12_5000_highlr(l,1) = sqrt(sum((DoubleNotchCoarse_enonlocal_true_vals - Adam_Predictions_DoubleNotchCoarse_12_12_5000_highlr(:,l)).^2 ./ DoubleNotchCoarse_enonlocal_true_vals.^2));
                clear current_file

                % Load file
                current_file = load(['Model11_TestE_DoubleNotchCoarse_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_140_v', num2str(runs(l)), '.txt']);
                Predictions_DoubleNotchCoarse_12_12_5000_highlr(:,l) = current_file / 10;
                DoubleNotchCoarse_l2norm_12_12_5000_highlr(l,1) = sqrt(sum((DoubleNotchCoarse_enonlocal_true_vals - Predictions_DoubleNotchCoarse_12_12_5000_highlr(:,l)).^2 ./ DoubleNotchCoarse_enonlocal_true_vals.^2));
                clear current_file

            end        
        end
    end
end

save Adam_Pred_L2norms_DoubleNotchCoarse_12_12_5000_highlr DoubleNotchCoarse_Adam_l2norm_12_12_5000_highlr
save LBFGS_Pred_L2norms_DoubleNotchCoarse_12_12_5000_highlr DoubleNotchCoarse_l2norm_12_12_5000_highlr




