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
layers  = [60];      % 8 12 20 32;
neurons = [7];      % 8 12 20 32;
epochs  = [5000];   
runs    = 1:10;
Predictions_StrC100_60_7_5000_lowlr       = zeros(10000,10);
Adam_Predictions_StrC100_60_7_5000_lowlr  = zeros(10000,10);

StrC100_l2norm_60_7_5000_lowlr        = zeros(length(runs),1);
StrC100_Adam_l2norm_60_7_5000_lowlr   = zeros(length(runs),1);

% -------------------------------------------------------------------------
StrC100_all_true_vals       = table2array(readtable("data_GP_StrC100_Nonlocal_gradient_Analytical_inc_164_init.csv"));
StrC100_enonlocal_true_vals = StrC100_all_true_vals(:,12);

% -------------------------------------------------------------------------
% Adam
for i = 1:length(layers)
    for j = 1:length(neurons)
        for k = 1:length(epochs)
            for l = 1:length(runs)

                % Load file
                current_file = load(['Model11_TestE_StrC100_Adam_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_164_v', num2str(runs(l)), '.txt']);
                Adam_Predictions_StrC100_60_7_5000_lowlr(:,l) = current_file / 10;
                StrC100_Adam_l2norm_60_7_5000_lowlr(l,1) = sqrt(sum((StrC100_enonlocal_true_vals - Adam_Predictions_StrC100_60_7_5000_lowlr(:,l)).^2 ./ StrC100_enonlocal_true_vals.^2));
                clear current_file

                % Load file
                current_file = load(['Model11_TestE_StrC100_Predictions_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_164_v', num2str(runs(l)), '.txt']);
                Predictions_StrC100_60_7_5000_lowlr(:,l) = current_file / 10;
                StrC100_l2norm_60_7_5000_lowlr(l,1) = sqrt(sum((StrC100_enonlocal_true_vals - Predictions_StrC100_60_7_5000_lowlr(:,l)).^2 ./ StrC100_enonlocal_true_vals.^2));
                clear current_file

            end        
        end
    end
end

save Adam_Pred_L2norms_StrC100_60_7_5000_lowlr StrC100_Adam_l2norm_60_7_5000_lowlr
save LBFGS_Pred_L2norms_StrC100_60_7_5000_lowlr StrC100_l2norm_60_7_5000_lowlr




