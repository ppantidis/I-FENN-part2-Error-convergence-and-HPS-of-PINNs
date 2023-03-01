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
layers  = [8];      % 8 10];
neurons = [8];     % 50 70];
epochs  = [10000];   % 10000];
runs    = 1:10;

for i = 1:length(layers)
    for j = 1:length(neurons)
        for k = 1:length(epochs)
            for l = 1:length(runs)
                % Load file
                current_file = load(['Model11_TestE_DoubleNotchCoarse_History_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_140_v', num2str(runs(l)), '.txt']);
                % Store cost values
                J_DoubleNotchCoarse_8_8_10000_highlr(l,:) = [current_file(epochs(k)) current_file(end)];
            end        
        end
    end
end


save Histories_DoubleNotchCoarse_8_8_10000_highlr J_DoubleNotchCoarse_8_8_10000_highlr






