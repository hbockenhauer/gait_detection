%% --- MStra Real-Time Gait Detection on Free-Living Stroke Patient Data ---
clear; clc; close all;

% --- 1. CONFIGURATION ---
dataPath = 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\Free_living';
plotPath = 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\mstraczkiewicz\MStraPlots_FreeLiving';

F_MIN    = 0.5;  F_MAX  = 3.50;
P_THRESH = 1.0;    A_THRESH = 0.08;
fs          = 50;
windowSize  = 2 * fs;   % 2s window
stepSize    = 1 * fs;   % 1s step
maxGap      = 1.5 / fs; % gap threshold for buffer reset

if ~exist(plotPath, 'dir'), mkdir(plotPath); end

% --- 2. INITIALIZE SUMMARY ---
summaryResults = table();

% --- 3. FIND ALL ANNOTATED FILES ---
allFiles = dir(fullfile(dataPath, '*_annotated.csv'));

fprintf('\nFound %d annotated files in: %s\n', length(allFiles), dataPath);
fprintf('%-30s | %-8s | %-8s | %-8s | %-8s\n', 'File', 'Accuracy', 'Precision', 'Recall', 'F1');
fprintf('%s\n', repmat('-', 1, 75));

for i = 1:length(allFiles)
    fileName  = allFiles(i).name;
    filePath  = fullfile(dataPath, fileName);

    % Extract subject name, e.g. Device2_sub1_annotated.csv -> sub1
    parts   = split(strrep(fileName, '_annotated.csv', ''), '_');
    subject = parts{2};

    try
        % --- A. LOAD DATA ---
        % Annotated files are comma-separated with a single 'time' column
        data = readtable(filePath, 'Delimiter', ',', 'VariableNamingRule', 'preserve');

        % --- B. PARSE TIMESTAMPS ---
        timeRaw = data{:, 'time'};
        if iscell(timeRaw), timeRaw = string(timeRaw); end
        fullDateTime = datetime(timeRaw, 'InputFormat', 'MM/dd/yyyy HH:mm:ss.SSS', 'Locale', 'en_US');

        % Remove duplicates and sort
        [~, uniqueIdx] = unique(fullDateTime, 'stable');
        data         = data(uniqueIdx, :);
        fullDateTime = fullDateTime(uniqueIdx);
        [fullDateTime, sortIdx] = sort(fullDateTime);
        data         = data(sortIdx, :);

        % Time vector in seconds
        time_vec = seconds(fullDateTime - fullDateTime(1));

        % --- C. EXTRACT ACCELEROMETER DATA ---
        % Free-living files use ax, ay, az column names
        ax = data{:, 'ax'};
        ay = data{:, 'ay'};
        az = data{:, 'az'};
        vm_all = sqrt(ax.^2 + ay.^2 + az.^2);

        % --- D. EXTRACT GROUND TRUTH LABELS ---
        labelIdx = find(strcmpi(data.Properties.VariableNames, 'Label'), 1);
        if ~isempty(labelIdx)
            y_true = data{:, labelIdx};
            if iscell(y_true) || isstring(y_true), y_true = str2double(y_true); end
        else
            fprintf('  Warning: No Label column in %s, skipping.\n', fileName);
            continue;
        end

        % --- E. REAL-TIME SIMULATION WITH GAP RESET ---
        totalSamples   = length(vm_all);
        y_pred_rt      = zeros(totalSamples, 1);
        circularBuffer = zeros(windowSize, 1);
        detectionState = 0;

        rt_T      = [];
        rt_peakF  = [];
        rt_maxPk  = [];
        rt_ampVal = [];

        for s = 2:totalSamples
            % Check for gap — reset buffer if too large
            dt = time_vec(s) - time_vec(s-1);
            if dt > maxGap
                circularBuffer(:) = 0;
                detectionState    = 0;
                continue;
            end

            % Update circular buffer
            circularBuffer = [circularBuffer(2:end); vm_all(s)];

            % Run detection every stepSize samples
            if mod(s, stepSize) == 0 && s >= windowSize
                [isGait, newState, m] = run_mstra_rt(...
                    circularBuffer, fs, F_MIN, F_MAX, P_THRESH, A_THRESH, detectionState);

                detectionState = newState;
                y_pred_rt(s - stepSize + 1 : s) = double(isGait);

                rt_T      = [rt_T,      time_vec(s)];
                rt_peakF  = [rt_peakF,  m.peakF];
                rt_maxPk  = [rt_maxPk,  m.maxPk];
                rt_ampVal = [rt_ampVal, m.ampVal];
            end
        end

        % --- F. COMPUTE METRICS ---
        evalIdx = windowSize:totalSamples;
        tp = sum(y_true(evalIdx) == 1 & y_pred_rt(evalIdx) == 1);
        tn = sum(y_true(evalIdx) == 0 & y_pred_rt(evalIdx) == 0);
        fp = sum(y_true(evalIdx) == 0 & y_pred_rt(evalIdx) == 1);
        fn = sum(y_true(evalIdx) == 1 & y_pred_rt(evalIdx) == 0);

        prec = tp / (tp + fp); if isnan(prec), prec = 0; end
        rec  = tp / (tp + fn); if isnan(rec),  rec  = 0; end
        f1   = 2 * (prec * rec) / (prec + rec); if isnan(f1), f1 = 0; end
        acc  = (tp + tn) / (tp + tn + fp + fn);

        fprintf('%-30s | %-8.3f | %-8.3f | %-8.3f | %-8.3f\n', fileName, acc, prec, rec, f1);

        summaryResults = [summaryResults; table({subject}, acc, f1, prec, rec, tp, tn, fp, fn, ...
            'VariableNames', {'Subject', 'Accuracy', 'F1', 'Precision', 'Recall', 'TP', 'TN', 'FP', 'FN'})];

        % --- G. PLOT ---
        try
            fig = figure('Name', fileName, 'Position', [50, 50, 1100, 850], ...
                         'Visible', 'off', 'Color', 'w');
            sgtitle(['MStra RT: ', strrep(fileName, '_', '\_')], 'FontSize', 13);

            color = '#0072BD';
            ax_h = zeros(4, 1);

            % Subplot 1: Frequency
            ax_h(1) = subplot(4, 1, 1); hold on;
            plot(rt_T, rt_peakF, 'Color', color, 'LineWidth', 1.2);
            yline([F_MIN, F_MAX], 'r--');
            ylabel('Freq (Hz)'); grid on;
            title('Criterion 1: Dominant Frequency');

            % Subplot 2: Power
            ax_h(2) = subplot(4, 1, 2); hold on;
            plot(rt_T, rt_maxPk, 'Color', color, 'LineWidth', 1.2);
            yline(P_THRESH, 'r--');
            ylabel('Power'); grid on;
            title('Criterion 2: Spectral Power');

            % Subplot 3: Amplitude
            ax_h(3) = subplot(4, 1, 3); hold on;
            plot(rt_T, rt_ampVal, 'Color', color, 'LineWidth', 1.2);
            yline(A_THRESH, 'r--');
            ylabel('Std Dev'); grid on;
            title('Criterion 3: Amplitude');

            % Subplot 4: GT vs Prediction
            ax_h(4) = subplot(4, 1, 4); hold on;
            area(time_vec, y_true, 'FaceColor', color, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            stairs(time_vec, y_pred_rt, 'Color', color, 'LineWidth', 1.5);
            ylabel('Gait (0/1)'); grid on;
            title(sprintf('Detection vs GT  |  Prec=%.2f  Rec=%.2f  F1=%.2f  Acc=%.2f', ...
                prec, rec, f1, acc));
            legend({'GT', 'Prediction'}, 'Location', 'northeastoutside');
            xlabel('Time (s)');

            linkaxes(ax_h, 'x');
            if ~isempty(rt_T)
                xlim(ax_h(1), [0, max(time_vec)]);
            end

            drawnow;
            saveName = strrep(fileName, '.csv', '_RT_Plot.png');
            saveas(fig, fullfile(plotPath, saveName));
            % fprintf('  --> Plot saved: %s\n', saveName);
            close(fig);

        catch ME_plot
            fprintf('  !! Plot error for %s: %s (line %d)\n', fileName, ME_plot.message, ME_plot.stack(1).line);
            if exist('fig', 'var') && ishandle(fig), close(fig); end
        end

    catch ME
        fprintf('  Error processing %s: %s\n', fileName, ME.message);
    end
end

% --- 4. SUMMARY ---
if ~isempty(summaryResults)
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('SUMMARY BY SUBJECT\n');
    fprintf('%s\n', repmat('=', 1, 60));
    subjectSummary = groupsummary(summaryResults, 'Subject', 'mean', {'Accuracy', 'Precision', 'Recall', 'F1'});
    disp(subjectSummary(:, [1, 3:6]));

    fprintf('\nOVERALL MEAN\n');
    fprintf('Accuracy:  %.3f\n', mean(summaryResults.Accuracy));
    fprintf('Precision: %.3f\n', mean(summaryResults.Precision));
    fprintf('Recall:    %.3f\n', mean(summaryResults.Recall));
    fprintf('F1:        %.3f\n', mean(summaryResults.F1));

    % Save results
    writetable(summaryResults, fullfile(plotPath, 'FreeLiving_MStra_Results.xlsx'), 'Sheet', 'All');
    writetable(subjectSummary, fullfile(plotPath, 'FreeLiving_MStra_Results.xlsx'), 'Sheet', 'By_Subject');
    fprintf('\nResults saved to FreeLiving_MStra_Results.xlsx\n');
end


%% --- RT DETECTION FUNCTION ---
function [finalDecision, newState, metrics] = run_mstra_rt(winData, fs, fMin, fMax, pThr, aThr, prevState)
    metrics.ampVal = std(winData);

    nfft    = 512;
    w       = hann(length(winData));
    winProc = (winData - mean(winData)) .* w;
    S       = fft(winProc, nfft);
    P       = abs(S(1:nfft/2+1)).^2;

    [metrics.maxPk, maxIdx] = max(P);
    freqs          = fs * (0:(nfft/2)) / nfft;
    metrics.peakF  = freqs(maxIdx);

    rawDecision   = (metrics.peakF >= fMin && metrics.peakF <= fMax && ...
                     metrics.maxPk > pThr && metrics.ampVal > aThr);

    % Require 2 consecutive detections (1 history element)
    newState      = rawDecision;
    finalDecision = prevState & rawDecision;
end