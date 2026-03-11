%% --- MStra Bayesian Hyperparameter Optimizer (QSense Data) ---
clear; clc; close all;
warning('off', 'MATLAB:table:ModifiedAndSavedVariableNames');

% --- 1. CONFIGURATION ---
% Point this to the root directory containing subject folders
dataPath = 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_mixed';
fs = 50;

% --- 2. DATA SPLITTING & PRE-LOADING ---
% QSense data is nested in subject folders. We find all valid txt files first.
subDirs = dir(dataPath);
subDirs = subDirs([subDirs.isdir] & ~ismember({subDirs.name}, {'.', '..'}));

allFilesInfo = [];
for i = 1:length(subDirs)
    folderPath = fullfile(dataPath, subDirs(i).name);
    % We target both left and right wrist files as separate training samples
    targets = {'s1_1RW.txt'; 's2_2LW.txt'};
    for t = 1:length(targets)
        fullFilePath = fullfile(folderPath, targets{t});
        if isfile(fullFilePath)
            allFilesInfo = [allFilesInfo; struct('path', fullFilePath, 'name', targets{t})];
        end
    end
end

if isempty(allFilesInfo), error('No QSense files found!'); end

% Shuffle and Split (70% Train, 30% Test)
rng(42); 
idx = randperm(length(allFilesInfo));
trainIdx = idx(1:round(0.7 * length(allFilesInfo)));
testIdx  = idx(round(0.7 * length(allFilesInfo))+1 : end);

fprintf('Pre-loading %d QSense training files...\n', length(trainIdx));
trainingSet = pre_load_qsense_data(allFilesInfo(trainIdx));

% --- 3. DEFINE SEARCH SPACE ---
vars = [
    optimizableVariable('F_MIN', [0.01, 0.9]),
    optimizableVariable('F_MAX', [2.0, 10.0]),
    
    % Power limits
    optimizableVariable('P_MIN', [0.1, 10.0], 'Transform', 'log'), 
    optimizableVariable('P_MAX', [10.0, 500.0], 'Transform', 'log'), 
    
    % Amplitude (Energy) limits
    optimizableVariable('A_MIN', [0.01*9.81^3, 0.2*9.81^3]), 
    optimizableVariable('A_MAX', [0.2*9.81^3, 2.0*9.81^3]) 
];

% --- 4. RUN OPTIMIZATION ---
fprintf('Starting Bayesian Optimization...\n');
objFunc = @(p) 1 - run_eval_iteration(p, trainingSet, fs);

results = bayesopt(objFunc, vars, ...
    'MaxObjectiveEvaluations', 100, ...
    'IsObjectiveDeterministic', true, ...
    'PlotFcn', {@plotMinObjective}); 

% --- 5. RESULTS & TEST ---
bestParams = results.XAtMinObjective;
fprintf('\nEvaluating Best Parameters on Test Set...\n');
testSet = pre_load_qsense_data(allFilesInfo(testIdx));
testF1  = run_eval_iteration(bestParams, testSet, fs);

fprintf('\nFinal Test Mean F1-Score: %.4f\n', testF1);
disp('Best Thresholds Found:');
disp(bestParams);

%% --- QSense Data Pre-Loader ---
function dataset = pre_load_qsense_data(filesInfo)
    dataset = cell(length(filesInfo), 1);
    fs = 50; % Define locally to keep function self-contained
    
    for i = 1:length(filesInfo)
        try
            opts = detectImportOptions(filesInfo(i).path);
            opts.VariableNamingRule = 'preserve';
            opts = setvartype(opts, [1, 2], 'string');
            data = readtable(filesInfo(i).path, opts);

            % --- STEP 1: PARSE TIMESTAMPS ---
            dateTimeStr = string(data{:,1}) + " " + string(data{:,2});
            fullDateTime = datetime(dateTimeStr, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');

            % --- STEP 0.5: REMOVE BACKWARDS-JUMP BLOCKS ---
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
            fullDateTime = fullDateTime(keepMask);
            data         = data(keepMask, :);

            % --- STEP 2: FIX TIME TRAVELERS (1970 / 2034 jumps) FIRST ---
            time_diffs = diff(fullDateTime);
            jumpIdx = find(abs(time_diffs) > days(100));
            for j = 1:length(jumpIdx)
                idx = jumpIdx(j);
                false_gap = time_diffs(idx) - seconds(1/fs);
                fullDateTime(idx+1:end) = fullDateTime(idx+1:end) - false_gap;
                time_diffs = diff(fullDateTime); % Recompute for next iteration
            end

            % --- STEP 3: GLOBAL SORT ---
            [fullDateTime, sIdx] = sort(fullDateTime);
            data = data(sIdx, :);

            % --- STEP 4: REMOVE DUPLICATE TIMESTAMPS ---
            [fullDateTime, uIdx] = unique(fullDateTime);
            data = data(uIdx, :);

            % --- STEP 5: BUILD TIME VECTOR ---
            s.time_vec = seconds(fullDateTime - fullDateTime(1));

            % --- STEP 6: EXTRACT SIGNALS ---
            s.vm     = sqrt(data{:,6}.^2 + data{:,7}.^2 + data{:,8}.^2);
            s.energy = data{:,13};

            % --- STEP 7: EXTRACT LABELS ---
            labelIdx = find(strcmpi(data.Properties.VariableNames, 'Label'), 1);
            if ~isempty(labelIdx)
                y = data{:, labelIdx};
                if iscell(y) || isstring(y)
                    y = str2double(y);
                end
                y(isnan(y)) = 0;
                s.y_true = double(y);
            else
                % Folder-level label: infer from filename path
                isGait = contains(lower(filesInfo(i).path), ["walk", "stairs"]);
                s.y_true = double(isGait) * ones(height(data), 1);
            end

            % --- STEP 8: SANITY CHECK ---
            % Ensure all signals are the same length after cleaning
            minLen = min([length(s.vm), length(s.energy), length(s.y_true)]);
            s.vm     = s.vm(1:minLen);
            s.energy = s.energy(1:minLen);
            s.y_true = s.y_true(1:minLen);
            s.time_vec = s.time_vec(1:minLen);

            dataset{i} = s;

        catch ME
            fprintf('  WARNING: Skipping file %s\n  Reason: %s\n', ...
                    filesInfo(i).path, ME.message);
            dataset{i} = []; % Leave empty; run_eval_iteration should skip []
        end
    end
end

function globalF1 = run_eval_iteration(p, dataset, fs)
    % Short-circuit impossible parameter combinations
    if (p.F_MIN >= p.F_MAX) || (p.P_MIN >= p.P_MAX) || (p.A_MIN >= p.A_MAX)
        globalF1 = 0;
        return;
    end

    % Accumulate counts globally across all files
    total_tp = 0;
    total_fp = 0;
    total_fn = 0;

    winSize = 2 * fs;
    step    = 1 * fs;
    maxGap  = 1.5 / fs;

    for i = 1:length(dataset)
        d = dataset{i};
        if isempty(d), continue; end  % Guard for files that failed to load

        y_pred = zeros(length(d.vm), 1);
        buffer = ones(winSize, 1) * d.vm(1);  % initialisation
        state  = 0;

        for s = 2:length(d.vm)
            if (d.time_vec(s) - d.time_vec(s-1)) > maxGap
                buffer(:) = d.vm(s); state = 0; continue; % gap reset
            end

            buffer = [buffer(2:end); d.vm(s)];

            if mod(s, step) == 0 && s >= winSize
                ampVal = d.energy(s);
                nfft   = 512;
                w      = hann(winSize);
                winProc = (buffer - mean(buffer)) .* w;

                S = fft(winProc, nfft);
                P = abs(S(1:nfft/2+1)).^2;

                [maxPk, maxIdx] = max(P);
                peakF = (fs * (maxIdx-1)) / nfft;

                rawDec = (peakF >= p.F_MIN && peakF <= p.F_MAX && ...
                          maxPk >  p.P_MIN  && maxPk  <= p.P_MAX && ...
                          ampVal > p.A_MIN  && ampVal <= p.A_MAX);

                isGait = (state & rawDec);
                state  = rawDec;

                idxRange = max(1, s-step+1) : s;
                y_pred(idxRange) = double(isGait);
            end
        end

        % Accumulate raw counts (skip burn-in window)
        sCount      = 0;
        sampleValid = false(length(d.vm), 1);
        for k = 2:length(d.vm)
            if (d.time_vec(k) - d.time_vec(k-1)) > maxGap
                sCount = 0;
            else
                sCount = sCount + 1;
            end
            sampleValid(k) = sCount >= winSize;  % windowSize in free-living optimizer
        end
        evalIdx = find(sampleValid)';
        if isempty(evalIdx), continue; end

        y_true_eval = d.y_true(evalIdx);
        y_pred_eval = y_pred(evalIdx);

        total_tp = total_tp + sum(y_true_eval == 1 & y_pred_eval == 1);
        total_fp = total_fp + sum(y_true_eval == 0 & y_pred_eval == 1);
        total_fn = total_fn + sum(y_true_eval == 1 & y_pred_eval == 0);
    end

    % Compute global F1 from pooled counts
    precision = total_tp / (total_tp + total_fp);
    recall    = total_tp / (total_tp + total_fn);

    if (precision + recall) == 0
        globalF1 = 0;
    else
        globalF1 = 2 * (precision * recall) / (precision + recall);
    end

    % Safeguard: if all predictions are zero (no gait detected at all),
    % precision would be NaN (0/0). Treat as 0.
    if isnan(globalF1), globalF1 = 0; end
end