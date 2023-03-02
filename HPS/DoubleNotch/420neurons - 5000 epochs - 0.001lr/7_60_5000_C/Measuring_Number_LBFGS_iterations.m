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
layers          = [7];
neurons         = [60];
epochs          = [5000];  
runs            = 1:10;

for i = 1:length(layers)
    for j = 1:length(neurons)
        for k = 1:length(epochs)
            for l = 1:length(runs)

                % Load file
                current_file = load(['Model11_TestE_DoubleNotchCoarse_History_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_140_v', num2str(runs(l)), '.txt']);
                
                % Store cost values
                No_LBFGS_iters_DNCoare_7_60_5000_highlr(l,1) = length(current_file) - epochs;
                
            end        
        end
    end
end

save No_LBFGS_iters_DNCoare_7_60_5000_highlr No_LBFGS_iters_DNCoare_7_60_5000_highlr








