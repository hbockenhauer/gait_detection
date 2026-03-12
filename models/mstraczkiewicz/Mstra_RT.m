%% --- Batch Real-Time Performance Evaluator with Debug Plots ---
clear; clc; close all;

% --- 1. CONFIGURATION ---
dataPaths = {
    % 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_mixed'
    % 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_edge'
    % 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data'
    'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_tests'
};
PlotPath  = 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\mstraczkiewicz\MStraPlots_RT';
% 0.044467    6.2058     2.647       103.4
% 0.0345    9.5063    2.6053    489.33    111.02    536.61
% 0.1169    6.8252    2.666    491.22    116.67    1059.8
% 0.097033    7.1325    2.3599    496.76    120.77    1105.1
F_MIN = 0.097033; F_MAX = 7.1325;
P_MIN = 2.3599; P_MAX = 496.76;
A_MIN = 120.77; A_MAX = 1105.1;
fs = 50;
windowSize = 2 * fs;
stepSize = 1 * fs;

if ~exist(PlotPath, 'dir'), mkdir(PlotPath); end

% --- 2. INITIALIZE SUMMARIES ---
summaryResults = table();
dataQuality = table();

% --- Before the for loop ---
allExecTimes = [];

for d = 1:length(dataPaths)
    dataPath = dataPaths{d};
    if ~exist(dataPath, 'dir'), continue; end
    
    subDirs = dir(dataPath);
    subDirs = subDirs([subDirs.isdir] & ~ismember({subDirs.name}, {'.', '..'}));
    
    fprintf('\nProcessing: %s\n', dataPath);
    fprintf('%-25s | %-8s | %-8s | %-8s | %-8s\n', 'Subject_Wrist', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
    fprintf('--------------------------------------------------------------------------------\n');
    
    for i = 1:length(subDirs)
        folderName = subDirs(i).name;
        folderPath = fullfile(dataPath, folderName);
        targetFiles = {'s0_Hub.txt', 'Right'; 's2_2LW.txt', 'Left'};
        
        plotData = struct(); 
        for f = 1:size(targetFiles, 1)
            fileName = targetFiles{f, 1};
            sideLabel = targetFiles{f, 2};
            fullFilePath = fullfile(folderPath, fileName);
            if ~isfile(fullFilePath), continue; end
            
            try
            % --- A. LOAD & CLEAN (Stitching Version) ---
                opts = detectImportOptions(fullFilePath);
                opts.VariableNamingRule = 'preserve';
                opts = setvartype(opts, [1, 2], 'string'); 
                data = readtable(fullFilePath, opts);
                
                dateTimeStr = string(data{:,1}) + " " + string(data{:,2});
                fullDateTime = datetime(dateTimeStr, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
                rowsBefore = height(data);

                % --- STEP 0: REMOVE BACKWARDS-JUMP BLOCKS ---
                % The device re-dumps its circular buffer, creating blocks that go
                % backwards in time by seconds. Drop any sample whose timestamp is
                % earlier than the running maximum seen so far.
                runningMax = fullDateTime(1);
                keepMask   = true(length(fullDateTime), 1);
                for k = 1:length(fullDateTime)
                    if fullDateTime(k) < runningMax
                        keepMask(k) = false;
                    else
                        runningMax = fullDateTime(k);
                    end
                end

                % --- DIAGNOSTIC: Print removed row indices and timestamps ---
                removedIdx = find(~keepMask);
                if ~isempty(removedIdx)
                    fprintf('    Removed rows (backwards filter):\n');
                    % Group into contiguous blocks for cleaner output
                    blockStarts = removedIdx([true; diff(removedIdx) > 1]);
                    blockEnds   = removedIdx([diff(removedIdx) > 1; true]);
                    for b = 1:length(blockStarts)
                        if blockStarts(b) == blockEnds(b)
                            fprintf('      Row %d: %s\n', blockStarts(b), string(fullDateTime(blockStarts(b))));
                        else
                            fprintf('      Rows %d-%d: %s to %s (%d rows)\n', ...
                                blockStarts(b), blockEnds(b), ...
                                string(fullDateTime(blockStarts(b))), ...
                                string(fullDateTime(blockEnds(b))), ...
                                blockEnds(b) - blockStarts(b) + 1);
                        end
                    end
                end
                
                fullDateTime = fullDateTime(keepMask);
                data         = data(keepMask, :);
                rowsAfterBackwards = height(data);
                
                % --- 1. FIX TIME TRAVELERS (1960, 2034) ---
                % Calculate time differences based on raw file order
                time_diffs = diff(fullDateTime);
                
                % Find indices where the jump is absurdly large (e.g., > 100 days)
                jumpIdx = find(abs(time_diffs) > days(100)); 
                
                for j = 1:length(jumpIdx)
                    idx = jumpIdx(j);
                    % Calculate the "false" gap, leaving a standard 0.02s step
                    false_gap = time_diffs(idx) - seconds(1/fs); 
                    
                    % Shift all subsequent timestamps back (or forward) to reality
                    fullDateTime(idx+1:end) = fullDateTime(idx+1:end) - false_gap;
                    
                    % Update diffs for the next loop in case there are multiple jumps
                    time_diffs = diff(fullDateTime); 
                end
                
                % --- 2. GLOBAL SORT ---
                % Now safe to sort, which fixes the millisecond out-of-order blocks
                [fullDateTime, sortIdx] = sort(fullDateTime);
                data = data(sortIdx, :);
                
                % --- 3. REMOVE REPEATS ---
                % Drops identical timestamps (buffer overwrites)
                [fullDateTime, uniqueIdx] = unique(fullDateTime);
                data = data(uniqueIdx, :);
                rowsAfterUnique = height(data); 

                fprintf('  [%s %s] Faulty rows removed: %d total \n', ...
                    folderName, sideLabel, ...
                    rowsBefore - rowsAfterUnique)
                
                % --- 4. CREATE LITERAL TIME VECTOR ---
                time_vec = seconds(fullDateTime - fullDateTime(1));
                
                % --- 5. Extract rest of data ---
                vm_all = sqrt(data{:,6}.^2 + data{:,7}.^2 + data{:,8}.^2);
                energy = data{:,13};
                
                % --- B. GROUND TRUTH EXTRACTION (COLUMN OR FOLDER) ---
                varNames = data.Properties.VariableNames;
                labelIdx = find(strcmpi(varNames, 'Label'), 1);
                
                if ~isempty(labelIdx)
                    % CASE 1: Sample-level label column exists (Annotated files)
                    raw_gt = data{:, labelIdx};
                    
                    % Convert to numeric if it's a string/cell
                    if iscell(raw_gt) || isstring(raw_gt)
                        raw_gt = str2double(raw_gt);
                    end
                    
                    % Clean up NaNs and ensure double
                    raw_gt(isnan(raw_gt)) = 0;
                    y_true = double(raw_gt);
                else
                    % CASE 2: Folder-level activity (Unannotated files)
                    % If "walk" or "stairs" is in the folder name, assume 100% gait
                    isGaitActivity = contains(lower(folderName), ["walk", "stairs"]);
                    y_true = double(isGaitActivity) * ones(height(data), 1);
                end

                % Verify size consistency
                if length(y_true) ~= length(vm_all)
                     y_true = y_true(1:length(vm_all)); 
                end

                % --- C. REAL-TIME SIMULATION WITH GAP RESET ---
                totalSamples = length(vm_all);
                y_pred_rt = zeros(totalSamples, 1);
                circularBuffer = ones(windowSize, 1) * vm_all(1);
                detectionState = 0;
                samplesSinceReset = 0;
                
                % INITIALIZE HERE (Once per file/wrist side)
                rt_T = []; 
                rt_peakF = []; 
                rt_maxPk = []; 
                rt_ampVal = [];
                rt_validWindow = [];

                % Threshold for a "Gap"
                maxGap = 1.5 / fs; 
                
                for s = 2:totalSamples
                    % 1. CHECK FOR GAPS
                    dt = time_vec(s) - time_vec(s-1);
                    
                    if dt > maxGap
                        circularBuffer(:) = vm_all(s);
                        detectionState = 0;
                        samplesSinceReset = 0;
                        continue;
                    end

                    samplesSinceReset = samplesSinceReset + 1;
                                        
                    % 2. Update Buffer
                    circularBuffer = [circularBuffer(2:end); vm_all(s)];
                    
                    % 3. Run Detection
                    if mod(s, stepSize) == 0 && s >= windowSize
                        [isGait, newState, m] = run_mstra_fast_rt_with_metrics(...
                            circularBuffer, fs, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX, detectionState, energy(s));
                        
                        detectionState = newState;
                        y_pred_rt(s-stepSize+1 : s) = double(isGait);
                        
                        % Store plot metrics
                        rt_T = [rt_T, time_vec(s)];
                        rt_peakF = [rt_peakF, m.peakF];
                        rt_maxPk = [rt_maxPk, m.maxPk];
                        rt_ampVal = [rt_ampVal, m.ampVal];
                        rt_validWindow = [rt_validWindow, samplesSinceReset >= windowSize];
                    end
                end

                plotData.(sideLabel).T_vec = rt_T;
                plotData.(sideLabel).peakF = rt_peakF;
                plotData.(sideLabel).maxPk = rt_maxPk;
                plotData.(sideLabel).ampVal = rt_ampVal;
                plotData.(sideLabel).validWindow = rt_validWindow;
                plotData.(sideLabel).time = time_vec;
                plotData.(sideLabel).y_pred = y_pred_rt;
                plotData.(sideLabel).y_true = y_true;
                plotData.(sideLabel).time_diffs = diff(time_vec);
                plotData.(sideLabel).time_for_diffs = time_vec(2:end);

                % Calculate Metrics
                % Build per-sample valid mask for metric calculation
                sCount = 0;
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
                
                tp = sum(y_true(evalIdx) == 1 & y_pred_rt(evalIdx) == 1);
                tn = sum(y_true(evalIdx) == 0 & y_pred_rt(evalIdx) == 0);
                fp = sum(y_true(evalIdx) == 0 & y_pred_rt(evalIdx) == 1);
                fn = sum(y_true(evalIdx) == 1 & y_pred_rt(evalIdx) == 0);
                
                prec = tp/(tp+fp); if isnan(prec), prec=1; end
                rec = tp/(tp+fn); if isnan(rec), rec=1; end
                f1 = 2*(prec*rec)/(prec+rec); if isnan(f1), f1=0; end
                acc = (tp+tn)/(tp+tn+fp+fn);
                
                fprintf('%-25s | %-8.2f | %-8.2f | %-8.2f | %-8.2f\n', ...
                        sprintf('%s_%s', folderName, sideLabel), acc, prec, rec, f1);

                % --- EXTRACTION OF SUBJECT AND ACTIVITY ---
                nameParts = split(folderName, '_');
                if length(nameParts) >= 2
                    activityType = nameParts{1};
                    subjectName  = nameParts{2};
                else
                    activityType = 'Unknown';
                    subjectName  = 'Unknown';
                end

                summaryResults = [summaryResults; table({subjectName}, {activityType}, {sideLabel}, acc, f1, prec, rec, tp, tn, fp, fn, ...
                    'VariableNames', {'Subject','Activity','Wrist','Accuracy','F1','Precision','Recall','TP','TN','FP','FN'})];

                % --- UPDATED DATA QUALITY CALCULATIONS ---
                totalRawRows = rowsBefore;
                faultyRowsRemoved = rowsBefore - rowsAfterUnique;
                pctFaultyRows = 100 * (faultyRowsRemoved / totalRawRows);
                
                totalCleanedSec = rowsAfterUnique / fs; % The amount of "good" data we have
                evaluableSec    = sum(sampleValid) / fs; % The amount of data the algorithm actually processed
                
                dataQuality = [dataQuality; table({subjectName}, {activityType}, {sideLabel}, ...
                    totalRawRows, faultyRowsRemoved, pctFaultyRows, totalCleanedSec, evaluableSec, ...
                    'VariableNames', {'Subject','Activity','Wrist','TotalRawRows','FaultyRowsRemoved','PctFaultyRows','TotalCleanedSec','EvaluableSec'})];
            catch ME
                fprintf('Error in %s: %s\n', folderName, ME.message);
            end
        end

        % --- 4. PAIRED PLOTTING ---
        if ~isempty(fieldnames(plotData))
            try
                fig = figure('Name', folderName, 'Position', [50, 50, 1100, 950], 'Visible', 'off', 'Color', 'w');
                sgtitle(['RT Causal Debug: ', folderName], 'Interpreter', 'none');
                colors = {'#0072BD', '#D95319'}; 
                sides = fieldnames(plotData);
                ax = zeros(5,1); 

                % Subplot 1: Timestamp gaps
                ax(1) = subplot(5,1,1); hold on;
                for si = 1:length(sides)
                    stem(plotData.(sides{si}).time_for_diffs, ...
                         plotData.(sides{si}).time_diffs, ...
                         'Color', colors{si}, 'MarkerSize', 1);
                end
                yline(maxGap, 'r--', 'Gap threshold');
                ylabel('\Delta t (s)'); grid on; title('Timestamp Gaps (red = reset)');

                % Subplot 2: Frequency (NaN inserted at invalid windows to break the line)
                ax(2) = subplot(5,1,2); hold on;
                for si = 1:length(sides)
                    validMask = logical(plotData.(sides{si}).validWindow);
                    T_plot = plotData.(sides{si}).T_vec;
                    F_plot = plotData.(sides{si}).peakF;
                    T_plot(~validMask) = NaN;
                    F_plot(~validMask) = NaN;
                    plot(T_plot, F_plot, 'Color', colors{si}, 'LineWidth', 1.2);
                end
                yline([F_MIN, F_MAX], 'r--'); ylabel('Freq (Hz)'); grid on; title('Crit 1: Frequency');

                % Subplot 3: Power (NaN inserted at invalid windows to break the line)
                ax(3) = subplot(5,1,3); hold on;
                for si = 1:length(sides)
                    validMask = logical(plotData.(sides{si}).validWindow);
                    T_plot = plotData.(sides{si}).T_vec;
                    P_plot = plotData.(sides{si}).maxPk;
                    T_plot(~validMask) = NaN;
                    P_plot(~validMask) = NaN;
                    plot(T_plot, P_plot, 'Color', colors{si});
                end
                yline([P_MIN, P_MAX], 'r--'); ylabel('Power'); grid on; title('Crit 2: Power');
          
                % Subplot 4: Amplitude (NaN inserted at invalid windows to break the line)
                ax(4) = subplot(5,1,4); hold on;
                for si = 1:length(sides)
                    validMask = logical(plotData.(sides{si}).validWindow);
                    T_plot = plotData.(sides{si}).T_vec;
                    A_plot = plotData.(sides{si}).ampVal;
                    T_plot(~validMask) = NaN;
                    A_plot(~validMask) = NaN;
                    plot(T_plot, A_plot, 'Color', colors{si});
                end
                yline([A_MIN, A_MAX], 'r--'); ylabel('StdDev'); grid on; title('Crit 3: Amplitude');

                % Subplot 5: Detection
                ax(5) = subplot(5,1,5); hold on;
                h_leg = [];
                for si = 1:length(sides)
                    y_true_plot = double(plotData.(sides{si}).y_true);
                    y_pred_plot = double(plotData.(sides{si}).y_pred);
                
                    % Build per-sample valid mask from time_vec
                    t = plotData.(sides{si}).time;
                    dt = diff(t);
                    sCount = 0;
                    sampleValid = false(length(t), 1);
                    for k = 2:length(t)
                        if dt(k-1) > maxGap
                            sCount = 0;
                        else
                            sCount = sCount + 1;
                        end
                        sampleValid(k) = sCount >= windowSize;
                    end
                    y_true_plot(~sampleValid) = NaN;
                    y_pred_plot(~sampleValid) = NaN;
                
                    a_h = area(t, y_true_plot, 'FaceColor', colors{si}, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
                    p_h = stairs(t, y_pred_plot, 'Color', colors{si}, 'LineWidth', 1.5);
                    h_leg = [h_leg, a_h, p_h];
                end

                if length(sides) == 2
                    legend([h_leg(1), h_leg(2), h_leg(3), h_leg(4)], {'GT R','Pred R','GT L','Pred L'}, ...
                        'Orientation', 'horizontal', 'Location', 'southoutside');
                else
                    legend({'GT','Pred'}, 'Orientation', 'horizontal', 'Location', 'southoutside');
                end
                ylabel('Gait (0/1)'); grid on; title('Final RT Decision');

                linkaxes(ax, 'x'); 
                if isfield(plotData.(sides{1}), 'T_vec') && ~isempty(plotData.(sides{1}).T_vec)
                    xlim(ax(1), [0 max(plotData.(sides{1}).time)]);
                end

                drawnow;
                savePath = fullfile(PlotPath, [folderName, '_RT_Plot.png']);
                saveas(fig, savePath);
                fprintf('  --> Plot saved: %s\n', savePath);
                close(fig);

            catch ME_plot
                fprintf('  !! PLOT ERROR for %s: %s\n', folderName, ME_plot.message);
                fprintf('     Line: %d\n', ME_plot.stack(1).line);
                if exist('fig','var') && ishandle(fig), close(fig); end
            end
        end
    end
end

% --- 5. AGGREGATED PERFORMANCE SUMMARIES (GLOBAL SUMS) ---
if ~isempty(summaryResults)
    statsToSum = {'TP', 'FP', 'TN', 'FN'};
    
    wristSum    = groupsummary(summaryResults, 'Wrist', 'sum', statsToSum);
    activitySum = groupsummary(summaryResults, 'Activity', 'sum', statsToSum);
    subjectSum  = groupsummary(summaryResults, 'Subject', 'sum', statsToSum);
    
    calcMetrics = @(t) addvars(t, ...
        t.sum_TP ./ (t.sum_TP + t.sum_FP), ...
        t.sum_TP ./ (t.sum_TP + t.sum_FN), ...
        (t.sum_TP + t.sum_TN) ./ (t.sum_TP + t.sum_TN + t.sum_FP + t.sum_FN), ...
        'NewVariableNames', {'Precision', 'Recall', 'Accuracy'});

    wristFinal    = calcMetrics(wristSum);
    activityFinal = calcMetrics(activitySum);
    subjectFinal  = calcMetrics(subjectSum);
    
    f1Func = @(p, r) 2 .* (p .* r) ./ (p + r);
    wristFinal.F1    = f1Func(wristFinal.Precision, wristFinal.Recall);
    activityFinal.F1 = f1Func(activityFinal.Precision, activityFinal.Recall);
    subjectFinal.F1  = f1Func(subjectFinal.Precision, subjectFinal.Recall);

    finalTables = {wristFinal, activityFinal, subjectFinal};
    for j = 1:3
        t = finalTables{j};
        t.Precision(isnan(t.Precision)) = 0;
        t.Recall(isnan(t.Recall)) = 0;
        t.F1(isnan(t.F1)) = 0;
        finalTables{j} = t;
    end
    [wristFinal, activityFinal, subjectFinal] = deal(finalTables{:});

    fprintf('\n======================================================================\n');
    fprintf('GLOBAL AGGREGATED METRICS (Total TP / Total Counts)\n');
    fprintf('----------------------------------------------------------------------\n');
    disp('BY WRIST:'); disp(wristFinal(:, [1, 6:9]));
    fprintf('----------------------------------------------------------------------\n');
    disp('BY ACTIVITY:'); disp(activityFinal(:, [1, 6:9]));
    fprintf('----------------------------------------------------------------------\n');
    disp('BY SUBJECT:'); disp(subjectFinal(:, [1, 6:9]));
    fprintf('======================================================================\n');

    total_tp = sum(summaryResults.TP);
    total_fp = sum(summaryResults.FP);
    total_fn = sum(summaryResults.FN);
    
    global_precision = total_tp / (total_tp + total_fp);
    global_recall    = total_tp / (total_tp + total_fn);
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall);
    
    fprintf('\nGLOBAL DATASET PERFORMANCE:\n');
    fprintf('Precision: %.4f | Recall: %.4f | F1: %.4f\n', ...
            global_precision, global_recall, global_f1);
        
    resultsFile = fullfile(PlotPath, 'Detailed_MStra_RT_Results.xlsx');
    writetable(summaryResults, resultsFile, 'Sheet', 'All_Files');
    writetable(wristFinal, resultsFile, 'Sheet', 'By_Wrist');
    writetable(activityFinal, resultsFile, 'Sheet', 'By_Activity');
    writetable(subjectFinal, resultsFile, 'Sheet', 'By_Subject');

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
    
    % Save to Excel
    writetable(dataQuality, resultsFile, 'Sheet', 'Data_Quality');
end

%% --- RT FUNCTION ---
function [finalDecision, newState, metrics] = run_mstra_fast_rt_with_metrics(winData, fs, fMin, fMax, pMin, pMax, aMin, aMax, prevState, energy)
    metrics.ampVal = energy;
    nfft = 512;
    w = hann(length(winData));
    winProc = (winData - mean(winData)) .* w;
    S = fft(winProc, nfft);
    P = abs(S(1:nfft/2+1)).^2;
    [metrics.maxPk, maxIdx] = max(P);
    freqs = fs*(0:(nfft/2))/nfft;
    metrics.peakF = freqs(maxIdx);
    
    rawDecision = (metrics.peakF >= fMin && metrics.peakF <= fMax && metrics.maxPk > pMin && metrics.maxPk < pMax && metrics.ampVal > aMin && metrics.ampVal < aMax);
    
    % Reduce to 2 consecutive seconds (1 history element) - F1 = 0.8589
    newState = rawDecision;
    finalDecision = prevState & rawDecision;
end