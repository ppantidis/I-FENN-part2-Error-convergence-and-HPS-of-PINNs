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
'DefaultLegendBox','on', 'DefaultLegendFontSize', 10, ...
'DefaultAxesBoxStyle', 'back', 'DefaultAxesBox', 'off')

% Load file
History_v4   = load(['Model11_TestE_StrC100_History_80hidlay_7units_10000ep_inc_164_v4.txt']);
History_v5   = load(['Model11_TestE_StrC100_History_80hidlay_7units_10000ep_inc_164_v5.txt']);
History_v6   = load(['Model11_TestE_StrC100_History_80hidlay_7units_10000ep_inc_164_v6.txt']);
History_v8   = load(['Model11_TestE_StrC100_History_80hidlay_7units_10000ep_inc_164_v8.txt']);

% Plot
f = figure("Position",[300 100 450 500]);
t = tiledlayout(1,1,'TileSpacing','compact','Padding','compact');
title0 = title(t, '\bf{Histories}');
set(title0,'interpreter','latex','fontsize', 14,'Color','k');

hold on; box on; grid on;
xticks([1e0 1e1 1e2 1e3 1e4]);
xlabel("Training epochs"); ylabel("J"); ylim([0.004 10]); xlim([1 2e4]);
set(gca, 'YMinorTick','on', 'YMinorGrid','on', 'YScale', 'log', 'XScale', 'log', 'fontsize', 13); 

plot(History_v4,'b');
plot(History_v5,'r');
plot(History_v6,'c');
plot(History_v8,'k');

legend(["80x7 - Model 4 (trivial)", "80x7 - Model 5 (trivial)", "80x7 - Model 6", "80x7 - Model 8"],'location','best');

print(f,'-dpdf', 'Trivial_Solution_Histories.pdf')














