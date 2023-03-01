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
layers  = [20];      % 8 12 20 32;
neurons = [20];      % 8 12 20 32;
epochs  = [10000];   
runs    = 1:10;
Predictions_StrI100_20_20_10000_highlr       = zeros(25600,10);
Adam_Predictions_StrI100_20_20_10000_highlr  = zeros(25600,10);

StrI100_l2norm_20_20_10000_highlr        = zeros(length(runs),1);
StrI100_Adam_l2norm_20_20_10000_highlr   = zeros(length(runs),1);

% -------------------------------------------------------------------------
StrI100_all_true_vals       = table2array(readtable("data_GP_StrI100_Nonlocal_gradient_Analytical_inc_164_init.csv"));
StrI100_enonlocal_true_vals = StrI100_all_true_vals(:,12);

% -------------------------------------------------------------------------
% Adam
for i = 1:length(layers)
    for j = 1:length(neurons)
        for k = 1:length(epochs)
            for l = 1:length(runs)

                % Load file
                current_file = load(['Model11_TestE_StrI100_Adam_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_164_v', num2str(runs(l)), '.txt']);
                Adam_Predictions_StrI100_20_20_10000_highlr(:,l) = current_file / 10;
                StrI100_Adam_l2norm_20_20_10000_highlr(l,1) = sqrt(sum((StrI100_enonlocal_true_vals - Adam_Predictions_StrI100_20_20_10000_highlr(:,l)).^2 ./ StrI100_enonlocal_true_vals.^2));
                clear current_file

                % Load file
                current_file = load(['Model11_TestE_StrI100_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_164_v', num2str(runs(l)), '.txt']);
                Predictions_StrI100_20_20_10000_highlr(:,l) = current_file / 10;
                StrI100_l2norm_20_20_10000_highlr(l,1) = sqrt(sum((StrI100_enonlocal_true_vals - Predictions_StrI100_20_20_10000_highlr(:,l)).^2 ./ StrI100_enonlocal_true_vals.^2));
                clear current_file

            end        
        end
    end
end

% save Adam_Pred_L2norms_StrI100_20_20_10000_highlr StrI100_Adam_l2norm_20_20_10000_highlr
% save LBFGS_Pred_L2norms_StrI100_20_20_10000_highlr StrI100_l2norm_20_20_10000_highlr




