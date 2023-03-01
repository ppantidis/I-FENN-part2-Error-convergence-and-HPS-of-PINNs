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
layers  = [8];      % 8 12 20 32;
neurons = [8];      % 8 12 20 32;
epochs  = [10000];   
runs    = 1:10;
Predictions_StrB100_8_8_10000_highlr       = zeros(6400,10);
Adam_Predictions_StrB100_8_8_10000_highlr  = zeros(6400,10);

StrB100_l2norm_8_8_10000_highlr        = zeros(length(runs),1);
StrB100_Adam_l2norm_8_8_10000_highlr   = zeros(length(runs),1);

% -------------------------------------------------------------------------
StrB100_all_true_vals       = table2array(readtable("data_GP_StrB100_Nonlocal_gradient_Analytical_inc_164_init.csv"));
StrB100_enonlocal_true_vals = StrB100_all_true_vals(:,12);

% -------------------------------------------------------------------------
% Adam
for i = 1:length(layers)
    for j = 1:length(neurons)
        for k = 1:length(epochs)
            for l = 1:length(runs)

                % Load file
                current_file = load(['Model11_TestE_StrB100_Adam_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_164_v', num2str(runs(l)), '.txt']);
                Adam_Predictions_StrB100_8_8_10000_highlr(:,l) = current_file / 10;
                StrB100_Adam_l2norm_8_8_10000_highlr(l,1) = sqrt(sum((StrB100_enonlocal_true_vals - Adam_Predictions_StrB100_8_8_10000_highlr(:,l)).^2 ./ StrB100_enonlocal_true_vals.^2));
                clear current_file

                % Load file
                current_file = load(['Model11_TestE_StrB100_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_164_v', num2str(runs(l)), '.txt']);
                Predictions_StrB100_8_8_10000_highlr(:,l) = current_file / 10;
                StrB100_l2norm_8_8_10000_highlr(l,1) = sqrt(sum((StrB100_enonlocal_true_vals - Predictions_StrB100_8_8_10000_highlr(:,l)).^2 ./ StrB100_enonlocal_true_vals.^2));
                clear current_file

            end        
        end
    end
end

save Adam_Pred_L2norms_StrB100_8_8_10000_highlr StrB100_Adam_l2norm_8_8_10000_highlr
save LBFGS_Pred_L2norms_StrB100_8_8_10000_highlr StrB100_l2norm_8_8_10000_highlr




