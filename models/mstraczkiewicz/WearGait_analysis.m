%% WearGait-PD: Dual-Wrist Global Summary
clear; clc; close all;

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
for k = 1:8
    hasModels = exist(fullfile(projectRoot, 'models'), 'dir');
    hasData = exist(fullfile(projectRoot, 'QSense_data'), 'dir') || ...
              exist(fullfile(projectRoot, 'Free_living'), 'dir') || ...
              exist(fullfile(projectRoot, 'WearGait-PD'), 'dir') || ...
              exist(fullfile(projectRoot, 'wisdm-dataset'), 'dir') || ...
              exist(fullfile(projectRoot, 'Datasets', 'QSense_data'), 'dir') || ...
              exist(fullfile(projectRoot, 'Datasets', 'Free_living'), 'dir') || ...
              exist(fullfile(projectRoot, 'Datasets', 'WearGait'), 'dir') || ...
              exist(fullfile(projectRoot, 'Datasets', 'wisdm-dataset'), 'dir');
    if hasModels && hasData
        break;
    end
    parentDir = fileparts(projectRoot);
    if strcmp(parentDir, projectRoot)
        break;
    end
    projectRoot = parentDir;
end

outputsRoot = fullfile(projectRoot, 'outputs');
resultsDir = fullfile(outputsRoot, 'results');
datasetName = 'WearGait';
dataCandidates = {
    fullfile(projectRoot, datasetName)
    fullfile(projectRoot, 'Datasets', datasetName)
    fullfile(projectRoot, 'Datasets', 'WearGait', datasetName)
};
dataCandidates = dataCandidates(cellfun(@(p) exist(p, 'dir') == 7, dataCandidates));
if ~isempty(dataCandidates)
    dataPath = dataCandidates{1};
else
    dataPath = fullfile(projectRoot, datasetName);
end

if ~exist(dataPath, 'dir')
    rootCandidates = {pwd, fileparts(pwd), fileparts(fileparts(pwd)), fileparts(fileparts(fileparts(pwd)))};
    foundDataPath = '';
    for r = 1:length(rootCandidates)
        rootCandidate = rootCandidates{r};
        candidatePaths = {
            fullfile(rootCandidate, datasetName)
            fullfile(rootCandidate, 'Datasets', datasetName)
            fullfile(rootCandidate, 'Datasets', 'WearGait', datasetName)
        };
        idx = find(cellfun(@(p) exist(p, 'dir') == 7, candidatePaths), 1);
        if ~isempty(idx)
            projectRoot = rootCandidate;
            foundDataPath = candidatePaths{idx};
            break;
        end
    end

    if ~isempty(foundDataPath)
        outputsRoot = fullfile(projectRoot, 'outputs');
        resultsDir = fullfile(outputsRoot, 'results');
        dataPath = foundDataPath;
        fprintf('Using fallback projectRoot from pwd: %s\n', projectRoot);
    else
        error('WearGait dataset directory not found: %s | projectRoot=%s | pwd=%s', dataPath, projectRoot, pwd);
    end
end

F_MIN = 0.50;
F_MAX = 3.50;
P_THRESH = 3;
A_THRESH = 0.1; 
fs = 100;

% --- 2. FILE INITIALIZATION ---
files = dir(fullfile(dataPath, '**', '*.csv'));
fileNames = lower(string({files.name}));
isWalkFile = contains(fileNames, 'freewalk');
isMetadata = contains(fileNames, 'manifest') | contains(fileNames, 'demographic');
files = files(isWalkFile & ~isMetadata);

if isempty(files)
    error('No WearGait walk CSV files found under: %s', dataPath);
end

summaryResults = table();
fprintf('Processing %d files (Checking both wrists)...\n', length(files));
fprintf('%-22s | %-8s | %-8s | %-8s | %-8s\n', 'Subject_Wrist', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
fprintf('--------------------------------------------------------------------------------\n');

% --- 3. MAIN PROCESSING LOOP ---
for i = 1:length(files)
    fileName = files(i).name;
    subjectID = strrep(fileName, '.csv', '');
    fullFilePath = fullfile(files(i).folder, fileName);
    
    try
        opts = detectImportOptions(fullFilePath);
        opts.VariableNamingRule = 'preserve';
        data = readtable(fullFilePath, opts);
        cols = data.Properties.VariableNames;
        
        % Identify available wrists (Right, Left, or Generic)
        wristPrefixes = {};
        if any(contains(cols, 'R_Wrist')), wristPrefixes{end+1} = 'R_Wrist_Acc_'; end
        if any(contains(cols, 'L_Wrist')), wristPrefixes{end+1} = 'L_Wrist_Acc_'; end
        
        % If no specific R/L labels, look for generic Acc
        if isempty(wristPrefixes) && any(contains(cols, 'Acc_X'))
             wristPrefixes{end+1} = 'Acc_'; 
        end

        % Process each wrist found
        for w = 1:length(wristPrefixes)
            prefix = wristPrefixes{w};
            sideLabel = strrep(prefix, '_Acc_', '');
            if isempty(sideLabel), sideLabel = 'Generic'; end
            
            % Extract Acceleration
            accX = data.([prefix, 'X']);
            accY = data.([prefix, 'Y']);
            accZ = data.([prefix, 'Z']);
            
            % Handle Time
            timeRaw = data.Time;
            if iscell(timeRaw), time = str2double(strrep(timeRaw, 'sec', ''));
            elseif isduration(timeRaw), time = seconds(timeRaw);
            else, time = timeRaw; end
            
            % Find Labels
            actCol = cols(contains(cols, 'Event') | contains(cols, 'Activity') | contains(cols, 'Label'));
            
            % Clean Data
            validRows = ~isnan(time) & ~isnan(accX) & ~isnan(accY) & ~isnan(accZ);
            timeClean = time(validRows);
            vm = sqrt(accX(validRows).^2 + accY(validRows).^2 + accZ(validRows).^2);
            
            % Create Ground Truth
            if isempty(actCol)
                y_true = zeros(size(vm));
            else
                labelData = string(data.(actCol{1})(validRows));
                y_true = contains(lower(labelData), ["walk", "stair", "gait", "jog", "free"]);
            end
            
            % Run Detection
            [y_pred, steps] = run_straczkiewicz_optimized(vm, fs, F_MIN, F_MAX, P_THRESH, A_THRESH);
            
            % Calculate Metrics
            tp = sum(y_true == 1 & y_pred == 1);
            tn = sum(y_true == 0 & y_pred == 0);
            fp = sum(y_true == 0 & y_pred == 1);
            fn = sum(y_true == 1 & y_pred == 0);
            
            acc = (tp + tn) / (tp + tn + fp + fn);
            prec = tp / (tp + fp); if isnan(prec), prec = 0; end
            rec = tp / (tp + fn); if isnan(rec), rec = 0; end
            f1 = 2 * (prec * rec) / (prec + rec); if isnan(f1), f1 = 0; end

            % Store results
            fullID = sprintf('%s_%s', subjectID, sideLabel);
            resRow = table({fullID}, {subjectID}, {sideLabel}, acc, prec, rec, f1, steps, ...
                'VariableNames', {'ID', 'Subject', 'Wrist', 'Accuracy', 'Precision', 'Recall', 'F1', 'Steps'});
            summaryResults = [summaryResults; resRow];
            
            fprintf('%-22s | %-8.2f | %-8.2f | %-8.2f | %-8.2f\n', ...
                    fullID, acc, prec, rec, f1);
        end

    catch ME
        fprintf('%-22s | ERROR: %s\n', subjectID, ME.message);
    end
end

% --- CALCULATE SUMMARY STATISTICS ---
% Overall Means
avgAcc  = mean(summaryResults.Accuracy);
avgPrec = mean(summaryResults.Precision);
avgRec  = mean(summaryResults.Recall);
avgF1   = mean(summaryResults.F1);

fprintf('\n======================================================================\n');
fprintf('GLOBAL PERFORMANCE SUMMARY (N=%d)\n', height(summaryResults));
fprintf('----------------------------------------------------------------------\n');
fprintf('Mean Accuracy:  %.2f\n', avgAcc);
fprintf('Mean Precision: %.2f\n', avgPrec);
fprintf('Mean Recall:    %.2f\n', avgRec);
fprintf('Mean F1-Score:  %.2f\n', avgF1);

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
resultsCsv = fullfile(resultsDir, 'sigpro_opt_WearGait_results.csv');
writetable(summaryResults, resultsCsv);
fprintf('Saved results to: %s\n', resultsCsv);

%% --- OPTIMIZED DETECTION FUNCTION ---
function [wi, steps, peakFs, ampVals, maxPks, T_vec] = run_straczkiewicz_optimized(vm, fs, fMin, fMax, pThr, aThr)
    fs_int = round(fs);
    [S, F, T_vec] = spectrogram(detrend(vm), 2*fs_int, fs_int, 512, fs);
    Cabs = abs(S).^2;
    numWindows = length(T_vec);
    peakFs = zeros(1, numWindows); maxPks = zeros(1, numWindows); ampVals = zeros(1, numWindows); wi_raw = zeros(1, numWindows);
    for i = 1:numWindows
        t_center = T_vec(i);
        idx = round(t_center * fs);
        win_idx = max(1, idx-fs_int):min(length(vm), idx+fs_int);
        ampVals(i) = std(vm(win_idx));
        [maxPks(i), maxIdx] = max(Cabs(:,i));
        peakFs(i) = F(maxIdx);
        if peakFs(i) >= fMin && peakFs(i) <= fMax && maxPks(i) > pThr && ampVals(i) > aThr
            wi_raw(i) = 1;
        end
    end
    wi_refined = movsum(wi_raw, [2 0]) >= 3;
    wi = zeros(size(vm));
    for i = 1:length(T_vec)
        if wi_refined(i)
            idx = round(T_vec(i) * fs);
            wi(max(1, idx-fs_int):min(length(wi), idx)) = 1;
        end
    end
    steps = sum(wi_refined);
end