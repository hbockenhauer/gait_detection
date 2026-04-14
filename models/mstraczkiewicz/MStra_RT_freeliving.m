%% --- MStra Real-Time Gait Detection on Free-Living Stroke Patient Data ---
clear; clc; close all;
warning('off', 'MATLAB:datetime:AmbiguousInputFormat');

% --- 1. CONFIGURATION ---
scriptFullPath = mfilename('fullpath');
if isempty(scriptFullPath)
    s = dbstack('-completenames');
    if ~isempty(s)
        scriptFullPath = s(1).file;
    end
end

if isempty(scriptFullPath)
    scriptDir = pwd;
else
    scriptDir = fileparts(scriptFullPath);
end

projectRoot = scriptDir;
for k = 1:6
    if exist(fullfile(projectRoot, 'Free_living'), 'dir') || ...
       exist(fullfile(projectRoot, 'Datasets', 'Free_living'), 'dir')
        break;
    end
    parentDir = fileparts(projectRoot);
    if strcmp(parentDir, projectRoot)
        break;
    end
    projectRoot = parentDir;
end

if ~exist(fullfile(projectRoot, 'Free_living'), 'dir') && ...
   ~exist(fullfile(projectRoot, 'Datasets', 'Free_living'), 'dir')
    if exist(fullfile(pwd, 'Free_living'), 'dir') || ...
       exist(fullfile(pwd, 'Datasets', 'Free_living'), 'dir')
        projectRoot = pwd;
    elseif exist(fullfile(fileparts(pwd), 'Free_living'), 'dir') || ...
           exist(fullfile(fileparts(pwd), 'Datasets', 'Free_living'), 'dir')
        projectRoot = fileparts(pwd);
    end
end

outputsRoot = fullfile(projectRoot, 'outputs');
resultsDir = fullfile(outputsRoot, 'results');
sigproResultsDir = fullfile(resultsDir, 'SigPro');
dataCandidates = {
    fullfile(projectRoot, 'Free_living')
    fullfile(projectRoot, 'Datasets', 'Free_living')
};
dataCandidates = dataCandidates(cellfun(@(p) exist(p, 'dir') == 7, dataCandidates));
if isempty(dataCandidates)
    error('Free_living dataset directory not found. projectRoot=%s | pwd=%s', projectRoot, pwd);
end
dataPath = dataCandidates{1};
datasetName = 'Free_living';
plotPath = fullfile(outputsRoot, 'plots', datasetName, 'SigPro');

% 0.96248    5.4359    0.78589    260.11    0.035257    0.12653
 % 0.89768    5.5618    0.83446    569.84    0.04083    0.28275
F_MIN = 0.96248;  F_MAX = 5.4359;
P_MIN = 0.78589;  P_MAX = 260.11;
A_MIN = 0.035257; A_MAX = 0.12653;
fs         = 50;
windowSize = 2 * fs;
stepSize   = 1 * fs;
maxGap     = 1.5 / fs;

if ~exist(plotPath, 'dir'), mkdir(plotPath); end
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
if ~exist(sigproResultsDir, 'dir'), mkdir(sigproResultsDir); end

% --- 2. INITIALIZE SUMMARY ---
summaryResults = table();
dataQuality    = table();

% --- 3. FIND ALL ANNOTATED FILES ---
allFiles = dir(fullfile(dataPath, '*_annotated.csv'));
if isempty(allFiles)
    allFiles = dir(fullfile(dataPath, '**', '*_annotated.csv'));
    if ~isempty(allFiles)
        fprintf('No top-level annotated files found. Using recursive search under: %s\n', dataPath);
    end
end

fprintf('\nFound %d annotated files in: %s\n', length(allFiles), dataPath);
fprintf('%-30s | %-8s | %-8s | %-8s | %-8s\n', 'File', 'Accuracy', 'Precision', 'Recall', 'F1');
fprintf('%s\n', repmat('-', 1, 75));

for i = 1:length(allFiles)
    fileName = allFiles(i).name;
    filePath = fullfile(allFiles(i).folder, fileName);

    % Extract subject name e.g. Device2_sub1_annotated.csv -> sub1
    parts   = split(strrep(fileName, '_annotated.csv', ''), '_');
    subject = parts{2};

    try
        % --- A. LOAD DATA ---
        opts = detectImportOptions(filePath, 'Delimiter', ',');
        opts.VariableNamingRule = 'preserve';
        opts = setvaropts(opts, 'time', 'InputFormat', 'MM/dd/yyyy HH:mm:ss.SSS');
        data = readtable(filePath, opts);

        % --- B. PARSE TIMESTAMPS ---
        timeRaw = data{:, 'time'};
        if iscell(timeRaw), timeRaw = string(timeRaw); end
        fullDateTime = datetime(timeRaw, 'InputFormat', 'MM/dd/yyyy HH:mm:ss.SSS', 'Locale', 'en_US');
        rowsBefore = height(data);

        % --- STEP 0: REMOVE BACKWARDS-JUMP BLOCKS ---
        runningMax = fullDateTime(1);
        keepMask   = true(length(fullDateTime), 1);
        for k = 1:length(fullDateTime)
            if fullDateTime(k) < runningMax
                keepMask(k) = false;
            else
                runningMax = fullDateTime(k);
            end
        end
        fprintf('  [%s] Removed %d samples (%.1fs) via backwards-block filter\n', ...
            fileName, sum(~keepMask), sum(~keepMask)/fs);
        fullDateTime = fullDateTime(keepMask);
        data         = data(keepMask, :);
        rowsAfterBackwards = height(data);

        % --- 1. FIX TIME TRAVELERS (>100 day jumps) ---
        time_diffs = diff(fullDateTime);
        jumpIdx    = find(abs(time_diffs) > days(100));
        for j = 1:length(jumpIdx)
            idx       = jumpIdx(j);
            false_gap = time_diffs(idx) - seconds(1/fs);
            fullDateTime(idx+1:end) = fullDateTime(idx+1:end) - false_gap;
            time_diffs = diff(fullDateTime);
        end

        % --- 2. GLOBAL SORT ---
        [fullDateTime, sortIdx] = sort(fullDateTime);
        data = data(sortIdx, :);

        % --- 3. REMOVE REPEATS ---
        [fullDateTime, uniqueIdx] = unique(fullDateTime);
        data = data(uniqueIdx, :);
        rowsAfterUnique = height(data); 

        fprintf('Faulty rows removed: %d total \n', ...
            rowsBefore - rowsAfterUnique)

        % --- 4. CREATE TIME VECTOR ---
        time_vec = seconds(fullDateTime - fullDateTime(1));

        % --- C. EXTRACT ACCELEROMETER DATA ---
        ax     = data{:, 'ax'};
        ay     = data{:, 'ay'};
        az     = data{:, 'az'};
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
        totalSamples      = length(vm_all);
        y_pred_rt         = zeros(totalSamples, 1);
        circularBuffer    = ones(windowSize, 1) * vm_all(1);
        detectionState    = 0;
        samplesSinceReset = 0;

        rt_T           = [];
        rt_peakF       = [];
        rt_maxPk       = [];
        rt_ampVal      = [];
        rt_validWindow = [];

        for s = 2:totalSamples
            dt = time_vec(s) - time_vec(s-1);
            if dt > maxGap
                circularBuffer(:) = vm_all(s);
                detectionState    = 0;
                samplesSinceReset = 0;
                continue;
            end

            samplesSinceReset = samplesSinceReset + 1;
            circularBuffer    = [circularBuffer(2:end); vm_all(s)];

            if mod(s, stepSize) == 0 && s >= windowSize
                [isGait, newState, m] = run_mstra_rt(...
                    circularBuffer, fs, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX, detectionState);

                detectionState = newState;
                y_pred_rt(s - stepSize + 1 : s) = double(isGait);

                rt_T           = [rt_T,           time_vec(s)];
                rt_peakF       = [rt_peakF,        m.peakF];
                rt_maxPk       = [rt_maxPk,        m.maxPk];
                rt_ampVal      = [rt_ampVal,        m.ampVal];
                rt_validWindow = [rt_validWindow,   samplesSinceReset >= windowSize];
            end
        end

        % --- F. BUILD VALID SAMPLE MASK FOR METRICS ---
        sCount      = 0;
        sampleValid = false(totalSamples, 1);
        for k = 2:totalSamples
            if (time_vec(k) - time_vec(k-1)) > maxGap
                sCount = 0;
            else
                sCount = sCount + 1;
            end
            sampleValid(k) = sCount >= windowSize;
        end
        evalIdx = find(sampleValid)';

        % --- G. COMPUTE METRICS ---
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
            'VariableNames', {'Subject','Accuracy','F1','Precision','Recall','TP','TN','FP','FN'})];

        % --- H. DATA QUALITY SUMMARY ---
        totalSec   = time_vec(end);
        validSec   = sum(sampleValid) / fs;
        removedSec = totalSec - validSec;
        removedPct = 100 * removedSec / totalSec;

        dataQuality = [dataQuality; table({subject}, {fileName}, ...
            rowsBefore, rowsBefore - rowsAfterUnique, ...
            totalSec, validSec, removedSec, removedPct, ...
            'VariableNames', {'Subject','File','TotalRawRows','FaultyRowsRemoved', ...
                              'TotalCleanedSec','EvaluableSec','RemovedSec','RemovedPct'})];

        % --- I. PLOT ---
        try
            fig = figure('Name', fileName, 'Position', [50, 50, 1100, 850], ...
                         'Visible', 'off', 'Color', 'w');
            sgtitle(['MStra RT: ', strrep(fileName, '_', '\_')], 'FontSize', 13);

            color  = '#0072BD';
            ax_h   = zeros(4, 1);

            % Subplot 1: Frequency (valid windows only)
            ax_h(1) = subplot(4, 1, 1); hold on;
            validMask = logical(rt_validWindow);
            T_plot = rt_T; F_plot = rt_peakF;
            T_plot(~validMask) = NaN; F_plot(~validMask) = NaN;
            plot(T_plot, F_plot, 'Color', color, 'LineWidth', 1.2);
            yline([F_MIN, F_MAX], 'r--');
            ylabel('Freq (Hz)'); grid on;
            title('Criterion 1: Dominant Frequency');

            % Subplot 2: Power (valid windows only)
            ax_h(2) = subplot(4, 1, 2); hold on;
            T_plot = rt_T; P_plot = rt_maxPk;
            T_plot(~validMask) = NaN; P_plot(~validMask) = NaN;
            plot(T_plot, P_plot, 'Color', color, 'LineWidth', 1.2);
            yline([P_MIN, P_MAX], 'r--');
            ylabel('Power'); grid on;
            title('Criterion 2: Spectral Power');

            % Subplot 3: Amplitude (valid windows only)
            ax_h(3) = subplot(4, 1, 3); hold on;
            T_plot = rt_T; A_plot = rt_ampVal;
            T_plot(~validMask) = NaN; A_plot(~validMask) = NaN;
            plot(T_plot, A_plot, 'Color', color, 'LineWidth', 1.2);
            yline([A_MIN, A_MAX], 'r--');
            ylabel('Std Dev'); grid on;
            title('Criterion 3: Amplitude');

            % Subplot 4: GT vs Prediction (valid samples only)
            ax_h(4) = subplot(4, 1, 4); hold on;
            y_true_plot = double(y_true);
            y_pred_plot = double(y_pred_rt);
            y_true_plot(~sampleValid) = NaN;
            y_pred_plot(~sampleValid) = NaN;
            area(time_vec, y_true_plot, 'FaceColor', color, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
            stairs(time_vec, y_pred_plot, 'Color', color, 'LineWidth', 1.5);
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
            close(fig);

        catch ME_plot
            fprintf('  !! Plot error for %s: %s (line %d)\n', fileName, ME_plot.message, ME_plot.stack(1).line);
            if exist('fig', 'var') && ishandle(fig), close(fig); end
        end

    catch ME
        fprintf('  Error processing %s: %s\n', fileName, ME.message);
    end
end

% --- 4. SUMMARY (AGGREGATED BY RAW COUNTS) ---
if ~isempty(summaryResults)
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('DETAILED SUMMARIES (GLOBAL AGGREGATION)\n');
    fprintf('%s\n', repmat('=', 1, 60));

    statsToSum = {'TP', 'FP', 'TN', 'FN'};
    subjectSum = groupsummary(summaryResults, 'Subject', 'sum', statsToSum);

    subjectSummary = addvars(subjectSum, ...
        subjectSum.sum_TP ./ (subjectSum.sum_TP + subjectSum.sum_FP), ...
        subjectSum.sum_TP ./ (subjectSum.sum_TP + subjectSum.sum_FN), ...
        (subjectSum.sum_TP + subjectSum.sum_TN) ./ (subjectSum.sum_TP + subjectSum.sum_TN + subjectSum.sum_FP + subjectSum.sum_FN), ...
        'NewVariableNames', {'Precision', 'Recall', 'Accuracy'});

    subjectSummary.F1 = 2 * (subjectSummary.Precision .* subjectSummary.Recall) ./ ...
                            (subjectSummary.Precision + subjectSummary.Recall);

    subjectSummary.Precision(isnan(subjectSummary.Precision)) = 0;
    subjectSummary.Recall(isnan(subjectSummary.Recall))       = 0;
    subjectSummary.F1(isnan(subjectSummary.F1))               = 0;

    disp(subjectSummary(:, {'Subject', 'Accuracy', 'Precision', 'Recall', 'F1'}));

    total_tp = sum(summaryResults.TP);
    total_fp = sum(summaryResults.FP);
    total_fn = sum(summaryResults.FN);
    total_tn = sum(summaryResults.TN);

    g_prec = total_tp / (total_tp + total_fp);
    g_rec  = total_tp / (total_tp + total_fn);
    g_f1   = 2 * (g_prec * g_rec) / (g_prec + g_rec);
    g_acc  = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn);

    fprintf('\nOVERALL DATASET TOTALS\n');
    fprintf('Accuracy:  %.3f\n', g_acc);
    fprintf('Precision: %.3f\n', g_prec);
    fprintf('Recall:    %.3f\n', g_rec);
    fprintf('F1:        %.3f\n', g_f1);

    fprintf('\n======================================================================\n');
    fprintf('DATA QUALITY SUMMARY\n');
    fprintf('----------------------------------------------------------------------\n');
    
    % Per-file summary
    disp(dataQuality);
    
    % --- 1. Aggregated Hardware/Buffer Data Loss ---
    globalTotalRawRows = sum(dataQuality.TotalRawRows);
    globalFaultyRows   = sum(dataQuality.FaultyRowsRemoved);
    globalFaultyPct    = 100 * (globalFaultyRows / globalTotalRawRows);
    
    % --- 2. Aggregated Algorithmic Uptime ---
    globalCleanedSec   = sum(dataQuality.TotalCleanedSec);
    globalEvaluableSec = sum(dataQuality.EvaluableSec);
    
    % Time lost to filling the buffer after startup or gaps
    globalBufferLostSec = globalCleanedSec - globalEvaluableSec; 
    globalEvaluablePct  = 100 * (globalEvaluableSec / globalCleanedSec);

    fprintf('--- 1. DATA CLEANING (Hardware & Buffer Drops) ---\n');
    fprintf('Total Raw Rows Read:   %d\n', globalTotalRawRows);
    fprintf('Faulty Rows Removed:   %d (%.2f%% of raw data)\n', globalFaultyRows, globalFaultyPct);
    
    fprintf('\n--- 2. ALGORITHM UPTIME (Post-Cleaning) ---\n');
    fprintf('Total Cleaned Time:    %.1f s\n', globalCleanedSec);
    fprintf('Evaluable Time:        %.1f s (%.1f%% of cleaned data)\n', globalEvaluableSec, globalEvaluablePct);
    fprintf('Time lost to gaps:     %.1f s (waiting for window to fill)\n', globalBufferLostSec);
    fprintf('======================================================================\n');
    
    % Save RT results to outputs/results
    resFile = fullfile(sigproResultsDir, 'sigpro_RT_Free_living_results.xlsx');
    writetable(summaryResults, resFile, 'Sheet', 'All_Files');
    writetable(subjectSummary, resFile, 'Sheet', 'By_Subject');
    writetable(dataQuality, resFile, 'Sheet', 'Data_Quality');
    fprintf('RT results saved to: %s\n', resFile);
end


%% --- RT DETECTION FUNCTION ---
function [finalDecision, newState, metrics] = run_mstra_rt(winData, fs, fMin, fMax, pMin, pMax, aMin, aMax, prevState)
    metrics.ampVal = std(winData);

    nfft    = 512;
    w       = hann(length(winData));
    winProc = (winData - mean(winData)) .* w;
    S       = fft(winProc, nfft);
    P       = abs(S(1:nfft/2+1)).^2;

    [metrics.maxPk, maxIdx] = max(P);
    freqs         = fs * (0:(nfft/2)) / nfft;
    metrics.peakF = freqs(maxIdx);

    rawDecision = (metrics.peakF >= fMin && metrics.peakF <= fMax && ...
                   metrics.maxPk >  pMin && metrics.maxPk <  pMax && ...
                   metrics.ampVal > aMin && metrics.ampVal < aMax);

        
    % two element history
    newState      = rawDecision;
    finalDecision = prevState & rawDecision;
end