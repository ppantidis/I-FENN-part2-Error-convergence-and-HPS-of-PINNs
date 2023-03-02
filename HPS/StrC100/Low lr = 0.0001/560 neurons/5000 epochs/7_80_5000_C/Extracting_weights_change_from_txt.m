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
layers  = [7];      % 8 10];
neurons = [80];     % 50 70];
epochs  = [5000];   % 10000];
runs    = 1:10;

for i = 1:length(layers)
    for j = 1:length(neurons)
        for k = 1:length(epochs)
            for l = 1:length(runs)
                % Load file
                Weights_Change_StrC100_7_80_5000_lowlr(l,:) = load(['Model11_TestE_StrC100_Weights_Change_', num2str(layers(i)), 'hidlay_', num2str(neurons(j)), 'units_', num2str(epochs(k)), 'ep_inc_164_v', num2str(runs(l)), '.txt']);
            end        
        end
    end
end


save Weights_Change_StrC100_7_80_5000_lowlr Weights_Change_StrC100_7_80_5000_lowlr






