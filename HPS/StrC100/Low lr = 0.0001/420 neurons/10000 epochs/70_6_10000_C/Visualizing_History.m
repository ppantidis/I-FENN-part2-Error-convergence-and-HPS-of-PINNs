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
History_v1   = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v1.txt']);
History_v2   = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v2.txt']);
History_v3   = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v3.txt']);
History_v4   = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v4.txt']);
History_v5   = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v5.txt']);
History_v6   = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v6.txt']);
History_v7   = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v7.txt']);
History_v8   = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v8.txt']);
History_v9   = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v9.txt']);
History_v10  = load(['Model11_TestE_StrC100_History_70hidlay_6units_10000ep_inc_164_v10.txt']);

% Plot
f = figure("Position",[300 100 720 500]);
t = tiledlayout(1,1,'TileSpacing','compact','Padding','compact');
title0 = title(t, '\bf{Histories (70x6 - 10000)}');
set(title0,'interpreter','latex','fontsize', 14,'Color','k');

hold on; box on; grid on;
xlabel("Training epochs"); ylabel("J"); ylim([1e-3 1e3]); xlim([0 2e4]);
set(gca, 'YMinorTick','on', 'YMinorGrid','on', 'YScale', 'log', 'XScale', 'log', 'fontsize', 13); 

plot(History_v1,'c');
plot(History_v2,'k');
plot(History_v3,'b');
plot(History_v4,'r');
plot(History_v5,'m');
plot(History_v6,'y');
plot(History_v7);
plot(History_v8);
plot(History_v9);
plot(History_v10);


legend(["M1","M2","M3","M4","M5","M6","M7","M8","M9","M10"],'location','best');

print(f,'-dpdf', 'All_Histories.pdf')














