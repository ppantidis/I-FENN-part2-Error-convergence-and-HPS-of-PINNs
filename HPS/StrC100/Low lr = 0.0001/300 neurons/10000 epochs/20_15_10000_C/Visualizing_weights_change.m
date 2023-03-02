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

% Load data
Weights_change_v1  = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v1.txt']);
Weights_change_v2  = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v2.txt']);
Weights_change_v3  = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v3.txt']);
Weights_change_v4  = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v4.txt']);
Weights_change_v5  = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v5.txt']);
Weights_change_v6  = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v6.txt']);
Weights_change_v7  = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v7.txt']);
Weights_change_v8  = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v8.txt']);
Weights_change_v9  = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v9.txt']);
Weights_change_v10 = load(['Model11_TestE_StrC100_Weights_Change_20hidlay_15units_10000ep_inc_164_v10.txt']);


% Plot
f = figure("Position",[300 100 720 500]);
t = tiledlayout(1,1,'TileSpacing','compact','Padding','compact');
title0 = title(t, '\bf{Histories (20x15 - 10000)}');
set(title0,'interpreter','latex','fontsize', 14,'Color','k');

hold on; box on; grid on;
ylabel("$\frac{|| \theta_{i} - \theta_{i - 1} ||_{2}}{|| \theta_{i - 1} ||_{2}}$");
xlabel("Training epochs (100s)"); ylim([5e-4 1e-1]); xlim([0 100]);
set(gca, 'YMinorTick','on', 'YMinorGrid','on', 'YScale', 'log', 'fontsize', 13); % 'YScale', 'log', 'XScale', 'log', 

plot(Weights_change_v1,'c');
plot(Weights_change_v2,'k');
plot(Weights_change_v3,'b');
plot(Weights_change_v4,'r');
plot(Weights_change_v5,'m');
plot(Weights_change_v6,'y');
plot(Weights_change_v7);
plot(Weights_change_v8);
plot(Weights_change_v9);
plot(Weights_change_v10);


legend(["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10"],'location','best');

print(f,'-dpdf', 'All_WeightsChange.pdf')






