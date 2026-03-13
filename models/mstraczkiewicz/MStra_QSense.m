clear; clc;
%% --- MStra detection on QSense wrist data (tuned) ---

fs = 50;                % Sample frequency
min_amp = 0.1;         % Minimum amplitude in g (lower to capture wrist swings)
T = 3;                  % Minimum walking duration (s)
delta = round(0.5 * fs);              % Local step peak window
alpha = 2;            % Min ratio below step frequency (allow small hand motion)
beta = 2;             % Max ratio above step frequency (ignore high harmonics)
step_freq = [0.5 3.5];  % Walking cadence frequency range (Hz)

% --- CONFIGURATION ---
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
qSenseCandidates = {
    fullfile(projectRoot, 'QSense_data_edge')
    fullfile(projectRoot, 'QSense_data')
    fullfile(projectRoot, 'QSense_data_mixed')
    fullfile(projectRoot, 'Datasets', 'QSense_data_edge')
    fullfile(projectRoot, 'Datasets', 'QSense_data')
    fullfile(projectRoot, 'Datasets', 'QSense_data_mixed')
};

dataPaths = qSenseCandidates(cellfun(@(p) exist(p, 'dir') == 7, qSenseCandidates));
dataPaths = unique(dataPaths, 'stable');
if isempty(dataPaths)
    rootCandidates = {pwd, fileparts(pwd), fileparts(fileparts(pwd)), fileparts(fileparts(fileparts(pwd)))};
    fallbackPaths = {};

    for r = 1:length(rootCandidates)
        rootCandidate = rootCandidates{r};
        candidatePaths = {
            fullfile(rootCandidate, 'QSense_data_edge')
            fullfile(rootCandidate, 'QSense_data')
            fullfile(rootCandidate, 'QSense_data_mixed')
            fullfile(rootCandidate, 'Datasets', 'QSense_data_edge')
            fullfile(rootCandidate, 'Datasets', 'QSense_data')
            fullfile(rootCandidate, 'Datasets', 'QSense_data_mixed')
        };
        existingPaths = candidatePaths(cellfun(@(p) exist(p, 'dir') == 7, candidatePaths));
        if ~isempty(existingPaths)
            projectRoot = rootCandidate;
            fallbackPaths = unique(existingPaths, 'stable');
            break;
        end
    end

    if ~isempty(fallbackPaths)
        outputsRoot = fullfile(projectRoot, 'outputs');
        resultsDir = fullfile(outputsRoot, 'results');
        dataPaths = fallbackPaths;
        fprintf('Using fallback projectRoot from pwd: %s\n', projectRoot);
    else
        error('No QSense dataset directories found. projectRoot=%s | pwd=%s', projectRoot, pwd);
    end
end

summaryResults = table();

for d = 1:length(dataPaths)
    dataPath = dataPaths{d};
    [~, datasetName] = fileparts(dataPath);
    fprintf('\nProcessing dataset: %s\n', dataPath);

    subDirs = dir(dataPath);
    subDirs = subDirs([subDirs.isdir] & ~ismember({subDirs.name}, {'.', '..'}));
    
    fprintf('%-30s | %-8s | %-8s | %-8s | %-8s\n', 'Subject_Wrist', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
    fprintf('--------------------------------------------------------------------------------\n');
    
    for i = 1:length(subDirs)
        folderName = subDirs(i).name;
        folderPath = fullfile(dataPath, folderName);
    
        targetFiles = {'s1_1RW.txt', 'Right'; 's2_2LW.txt', 'Left'};
    
        for t = 1:size(targetFiles, 1)
            fileName = targetFiles{t, 1};
            sideLabel = targetFiles{t, 2};
            fullFilePath = fullfile(folderPath, fileName);
    
            if ~isfile(fullFilePath), continue; end
    
            try
                % Load QSense Data
                opts = detectImportOptions(fullFilePath);
                opts.VariableNamingRule = 'preserve';
                
                % Ensure first two columns are read as text (date + time)
                if width(opts.VariableTypes) >= 2
                    opts.VariableTypes{1} = 'char';
                    opts.VariableTypes{2} = 'char';
                end
                
                data = readtable(fullFilePath, opts);               
              
                % Remove first 10s (500 rows) from data due to latency
                startRow = fs * 10;
                data = data(startRow:end, :);

                % Handle Time (Force 50Hz row-by-row)
                numRows = height(data);
                              
                % Create time vector: starts at 0, increments by 1/fs per row
                time = (0:numRows-1)' / fs;   

                % Extract acceleration
                accX = data{:, 6};
                accY = data{:, 7};
                accZ = data{:, 8};

                % Remove NaN
                validRows = ~isnan(accX) & ~isnan(accY) & ~isnan(accZ);
                accX = accX(validRows);
                accY = accY(validRows);
                accZ = accZ(validRows);

                % Vector magnitude (column vector)
                vm = sqrt(accX.^2 + accY.^2 + accZ.^2);
                vm = vm(:);

                if length(vm) < fs * T
                    continue;
                end

                % % --- BANDPASS FILTERING ---
                % [b,a] = butter(4,[0.5 4]/(fs/2),'bandpass');  % focus on walking frequencies
                % vm = filtfilt(b,a,vm);

                % Ground Truth Extraction
                % -------------------------------
                varNames = data.Properties.VariableNames;
                
                % CASE 1: Sample-level label column exists
                labelIdx = find(strcmpi(varNames, 'label'), 1);
                
                if ~isempty(labelIdx)
                    
                    raw_gt = data{:, labelIdx};
                    
                    % Convert to numeric (like pd.to_numeric)
                    if iscell(raw_gt) || isstring(raw_gt)
                        raw_gt = str2double(raw_gt);
                    end
                    
                    raw_gt(isnan(raw_gt)) = 0;
                    raw_gt = double(raw_gt);
                    
                    y_true = raw_gt(validRows);

                    % CASE 2: Folder-level activity
                else
                    isGaitActivity = contains(lower(folderName), ["walk","stairs"]);
                    y_true = double(isGaitActivity) * ones(size(vm));
                end

                % --- RUN MStra WALKING DETECTION ---
                [wi, steps, cad] = find_walking(vm, fs, min_amp, T, delta, alpha, beta, step_freq);

                wi = wi(:);
                if length(wi) ~= length(vm)
                    if length(wi) < length(vm)
                        wi = [wi; zeros(length(vm) - length(wi), 1)];
                    else
                        wi = wi(1:length(vm));
                    end
                end

                y_pred = wi;

                % --- METRICS ---
                tp = sum(y_true == 1 & y_pred == 1);
                tn = sum(y_true == 0 & y_pred == 0);
                fp = sum(y_true == 0 & y_pred == 1);
                fn = sum(y_true == 1 & y_pred == 0);

                total = tp + tn + fp + fn;
                if total == 0, continue; end

                acc = (tp + tn) / total;
                prec = tp / (tp + fp); if (tp + fp) == 0, prec = 1; end
                rec = tp / (tp + fn); if (tp + fn) == 0, rec = 1; end
                f1 = 2 * (prec * rec) / (prec + rec); if (prec + rec) == 0, f1 = 0; end

                steps_count = length(steps);

                % --- STORE ---
                fullID = sprintf('%s_%s', folderName, sideLabel);
                resRow = table({datasetName}, {fullID}, {folderName}, {sideLabel}, acc, prec, rec, f1, ...
                               steps_count, tp, tn, fp, fn, ...
                    'VariableNames', {'Dataset', 'ID', 'Subject', 'Wrist', 'Accuracy', ...
                                      'Precision', 'Recall', 'F1', 'Steps', ...
                                      'TP', 'TN', 'FP', 'FN'});
                summaryResults = [summaryResults; resRow];

                fprintf('%-30s | %8.4f | %8.4f | %8.4f | %8.4f\n', ...
                    fullID, acc, prec, rec, f1);

            catch ME
                fprintf('%-30s | ERROR: %s at line %d\n', ...
                    [folderName '_' sideLabel], ME.message, ME.stack(1).line);
            end
        end
    end
end

%% --- GLOBAL STATISTICS ---
if ~isempty(summaryResults)
    TP = sum(summaryResults.TP);
    TN = sum(summaryResults.TN);
    FP = sum(summaryResults.FP);
    FN = sum(summaryResults.FN);

    globalAcc = (TP + TN) / (TP + TN + FP + FN);
    globalPrec = TP / (TP + FP); if (TP + FP) == 0, globalPrec = 1; end
    globalRec = TP / (TP + FN); if (TP + FN) == 0, globalRec = 1; end
    globalF1 = 2 * (globalPrec * globalRec) / (globalPrec + globalRec);
    if (globalPrec + globalRec) == 0, globalF1 = 0; end

    fprintf('\nGLOBAL PERFORMANCE\n');
    fprintf('Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n', ...
        globalAcc, globalPrec, globalRec, globalF1);
end

%% --- EXPORT RESULTS ---
if ~isempty(summaryResults)
    if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
    csvFileName = fullfile(resultsDir, 'sigpro_MStra_QSense_results.csv');
    writetable(summaryResults, csvFileName);
    fprintf('Results saved to: %s\n', csvFileName);
end
