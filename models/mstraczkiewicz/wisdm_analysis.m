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
    hasData = exist(fullfile(projectRoot, 'Baseline'), 'dir') || ...
              exist(fullfile(projectRoot, 'Free_living'), 'dir') || ...
              exist(fullfile(projectRoot, 'WearGait-PD'), 'dir') || ...
              exist(fullfile(projectRoot, 'wisdm-dataset'), 'dir') || ...
              exist(fullfile(projectRoot, 'Datasets', 'Baseline'), 'dir') || ...
              exist(fullfile(projectRoot, 'Datasets', 'Free_living'), 'dir') || ...
              exist(fullfile(projectRoot, 'Datasets', 'WearGait', 'WearGait-PD'), 'dir') || ...
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
datasetName = 'wisdm-dataset';
dataCandidates = {
    fullfile(projectRoot, 'wisdm-dataset', 'raw', 'watch', 'accel')
    fullfile(projectRoot, 'Datasets', 'wisdm-dataset', 'raw', 'watch', 'accel')
};
dataCandidates = dataCandidates(cellfun(@(p) exist(p, 'dir') == 7, dataCandidates));
if ~isempty(dataCandidates)
    dataPath = dataCandidates{1};
else
    dataPath = fullfile(projectRoot, 'wisdm-dataset', 'raw', 'watch', 'accel');
end
saveResults = true; % Added missing variable

if ~exist(dataPath, 'dir')
    rootCandidates = {pwd, fileparts(pwd), fileparts(fileparts(pwd)), fileparts(fileparts(fileparts(pwd)))};
    foundDataPath = '';
    for r = 1:length(rootCandidates)
        rootCandidate = rootCandidates{r};
        candidatePaths = {
            fullfile(rootCandidate, 'wisdm-dataset', 'raw', 'watch', 'accel')
            fullfile(rootCandidate, 'Datasets', 'wisdm-dataset', 'raw', 'watch', 'accel')
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
        error('WISDM dataset directory not found: %s | projectRoot=%s | pwd=%s', dataPath, projectRoot, pwd);
    end
end

F_MIN = 0.5;
F_MAX = 3.50;
P_THRESH = 3;
A_THRESH = 0.1; 

% --- Straczkiewicz Parameters ---
fs = 20; 
min_amp   = 0.1;          % Minimum amplitude (g)
T         = 3;             % Minimum walking duration (s)
alpha     = 2;
beta      = 2;
step_freq = [0.5 3.5];     % Cadence range (Hz)
delta = round(0.5 * fs);

% --- 2. ACTIVITY MAPPING ---
gaitLabels = {'A', 'C'}; 


% --- 3. FILE INITIALIZATION ---
files = dir(fullfile(dataPath, '*.txt')); 
if isempty(files)
    error('No .txt files found. Check your dataPath.');
end

summaryResults = table();
fprintf('Processing %d WISDM files...\n', length(files));
fprintf('%-22s | %-8s | %-8s | %-8s | %-8s\n', 'Subject', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
fprintf('----------------------------------------------------------------------\n');

% --- 4. MAIN PROCESSING LOOP ---
for i = 1:length(files)
    fileName = files(i).name;
    subjectID = strrep(fileName, '.txt', '');
    
    try
        % A. Load Data
        opts = delimitedTextImportOptions("NumVariables", 6);
        opts.Delimiter = ",";
        opts.VariableNames = ["Subject", "Activity", "Time", "Acc_X", "Acc_Y", "Acc_Z"];
        opts.VariableTypes = ["double", "string", "double", "double", "double", "string"]; % Read Z as string first
        
        data = readtable(fullfile(dataPath, fileName), opts);
        
        % B. ROBUST CLEANING
        % 1. Clean Acc_Z (remove semicolon and convert to double)
        cleanZ = strrep(string(data.Acc_Z), ';', '');
        data.Acc_Z_Num = str2double(cleanZ);
        
        % 2. Remove any rows with NaN in critical columns
        validIdx = ~isnan(data.Time) & ~isnan(data.Acc_X) & ...
                   ~isnan(data.Acc_Y) & ~isnan(data.Acc_Z_Num);
        
        if sum(validIdx) < 100
            error('Not enough valid data points in file.');
        end
        
        cleanData = data(validIdx, :);
        
        % C. Pre-processing
        time = (cleanData.Time - cleanData.Time(1)) / 1e9; 
        vm = sqrt(cleanData.Acc_X.^2 + cleanData.Acc_Y.^2 + cleanData.Acc_Z_Num.^2);
        y_true = ismember(cleanData.Activity, gaitLabels);
        
        % D. Run Detection
        fs = round(1 / median(diff(time(1:min(500, end)))));
        %[y_pred, steps] = run_straczkiewicz_optimized(vm, fs, F_MIN, F_MAX, P_THRESH, A_THRESH);
        [wi, steps, cad] = find_walking(vm, fs, min_amp, T, ...
                                            delta, alpha, beta, step_freq);

        y_pred = wi(:);
        y_true = y_true(:);

        % --- ADD THIS ALIGNMENT BLOCK ---
        minLen = min(length(y_pred), length(y_true));
        y_pred = y_pred(1:minLen);
        y_true = y_true(1:minLen);
        % --------------------------------
        
        % E. Metrics
        tp = sum(y_true == 1 & y_pred == 1);
        tn = sum(y_true == 0 & y_pred == 0);
        fp = sum(y_true == 0 & y_pred == 1);
        fn = sum(y_true == 1 & y_pred == 0);
        
        acc = (tp + tn) / (tp + tn + fp + fn);
        prec = tp / (tp + fp); if isnan(prec), prec = 0; end
        rec = tp / (tp + fn); if isnan(rec), rec = 0; end
        f1 = 2 * (prec * rec) / (prec + rec); if isnan(f1), f1 = 0; end

        resRow = table({datasetName}, {subjectID}, acc, prec, rec, f1, steps, ...
            'VariableNames', {'Dataset', 'Subject', 'Accuracy', 'Precision', 'Recall', 'F1', 'Steps'});
        summaryResults = [summaryResults; resRow];
        
        fprintf('%-22s | %-8.2f | %-8.2f | %-8.2f | %-8.2f\n', ...
                subjectID, acc, prec, rec, f1);

    catch ME
        fprintf('%-22s | ERROR: %s\n', subjectID, ME.message);
    end
end

% Print Final Averages
if ~isempty(summaryResults)
    fprintf('----------------------------------------------------------------------\n');
    fprintf('%-22s | %-8.2f | %-8.2f | %-8.2f | %-8.2f\n', ...
        'AVERAGE', mean(summaryResults.Accuracy), mean(summaryResults.Precision), ...
        mean(summaryResults.Recall), mean(summaryResults.F1));
end

if saveResults && ~isempty(summaryResults)
    if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
    outCsv = fullfile(resultsDir, 'sigpro_MStra_WISDM_results.csv');
    writetable(summaryResults, outCsv);
    fprintf('Saved results to: %s\n', outCsv);
end

% function [wi, steps] = run_straczkiewicz_lite(vm, fs)
%     fs_int = round(fs);
%     vm_filt = detrend(vm); 
% 
%     [S, F, T_vec] = spectrogram(vm_filt, 2*fs_int, fs_int, 512, fs);
%     Cabs = abs(S).^2;
% 
%     wi_raw = zeros(size(T_vec));
%     for i = 1:length(T_vec)
%         [pks, locs] = findpeaks(Cabs(:,i), F);
%         if isempty(pks), continue; end
%         [~, maxIdx] = max(pks);
%         domFreq = locs(maxIdx);
%         if domFreq >= 0.6 && domFreq <= 3.4 && pks(maxIdx) > 0.0001
%             wi_raw(i) = 1;
%         end
%     end
%     wi_refined = movsum(wi_raw, [2 0]) >= 3;
%     wi = zeros(size(vm));
%     for i = 1:length(T_vec)
%         if wi_refined(i)
%             idx = round(T_vec(i) * fs);
%             wi(max(1, idx-fs_int):min(length(wi), idx)) = 1;
%         end
%     end
%     steps = sum(wi_refined);
% end