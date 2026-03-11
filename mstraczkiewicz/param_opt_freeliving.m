%% --- MStra Bayesian Hyperparameter Optimizer (Free-Living Data) ---
clear; clc; close all;
warning('off', 'MATLAB:datetime:AmbiguousInputFormat'); % Silence the format warnings

% --- 1. SETTINGS ---
dataPath = 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\Free_living';
fs = 50;

% --- 2. DATA SPLITTING & PRE-LOADING ---
allFiles = dir(fullfile(dataPath, '*_annotated.csv'));
if isempty(allFiles), error('No files found!'); end

% Shuffle and Split (70% Train, 30% Test)
rng(42); 
idx = randperm(length(allFiles));
trainIdx = idx(1:round(0.7 * length(allFiles)));
testIdx  = idx(round(0.7 * length(allFiles))+1 : end);

fprintf('Pre-loading %d training files...\n', length(trainIdx));
trainingSet = pre_load_data(allFiles(trainIdx), dataPath);

% --- 3. DEFINE OPTIMIZATION VARIABLES ---
vars = [
    optimizableVariable('F_MIN', [0.1, 1.5]),
    optimizableVariable('F_MAX', [2.0, 10.0]),
    
    % Power limits
    optimizableVariable('P_MIN', [0.1, 3.0], 'Transform', 'log'), 
    optimizableVariable('P_MAX', [10.0, 1000.0], 'Transform', 'log'), 
    
    % Amplitude (StdDev) limits
    optimizableVariable('A_MIN', [0.001, 0.1]), 
    optimizableVariable('A_MAX', [0.12, 1.0]) 
];

% --- 4. RUN BAYESIAN OPTIMIZATION ---
fprintf('Starting Bayesian Optimization...\n');
objFunc = @(p) 1 - run_eval_iteration(p, trainingSet, fs);

results = bayesopt(objFunc, vars, ...
    'MaxObjectiveEvaluations', 100, ...
    'IsObjectiveDeterministic', true, ...
    'PlotFcn', {@plotMinObjective}); 

% --- 5. EXTRACT BEST PARAMETERS ---
bestParams = results.XAtMinObjective;
fprintf('\n======================================\n');
fprintf('OPTIMIZATION COMPLETE\n');
disp(bestParams);
fprintf('======================================\n');

% --- 6. FINAL TEST (ON UNSEEN DATA) ---
fprintf('\nEvaluating on unseen TEST SET (%d subjects)...\n', length(testIdx));
testSet = pre_load_data(allFiles(testIdx), dataPath);
testF1  = run_eval_iteration(bestParams, testSet, fs);

fprintf('\n======================================\n');
fprintf('FINAL PERFORMANCE ON UNSEEN DATA\n');
fprintf('Mean F1-Score: %.4f\n', testF1);
fprintf('======================================\n');

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

    windowSize = 2 * fs;
    stepSize   = 1 * fs;
    maxGap     = 1.5 / fs;

    for i = 1:length(dataset)
        data = dataset{i};
        if isempty(data), continue; end  % Guard for files that failed to load

        vm       = data.vm;
        y_true   = data.y_true;
        time_vec = data.time_vec;

        y_pred         = zeros(length(vm), 1);
        circularBuffer = ones(windowSize, 1) * vm(1);
        detectionState = 0;

        for s = 2:length(vm)
            if (time_vec(s) - time_vec(s-1)) > maxGap
                circularBuffer(:) = vm(s); detectionState = 0; continue;
            end

            circularBuffer = [circularBuffer(2:end); vm(s)];

            if mod(s, stepSize) == 0 && s >= windowSize
                ampVal  = std(circularBuffer);
                nfft    = 512;
                w       = hann(windowSize);
                winProc = (circularBuffer - mean(circularBuffer)) .* w;

                S = fft(winProc, nfft);
                P = abs(S(1:nfft/2+1)).^2;

                [maxPk, maxIdx] = max(P);
                peakF = (fs * (maxIdx-1)) / nfft;

                rawDecision = (peakF >= p.F_MIN && peakF <= p.F_MAX && ...
                               maxPk  >  p.P_MIN && maxPk  <= p.P_MAX && ...
                               ampVal >  p.A_MIN && ampVal <= p.A_MAX);

                isGait         = (detectionState & rawDecision);
                detectionState = rawDecision;

                idxRange = max(1, s - stepSize + 1) : s;
                y_pred(idxRange) = double(isGait);
            end
        end

        % Accumulate raw counts (skip burn-in window)
        sCount      = 0;
        sampleValid = false(length(vm), 1);
        for k = 2:length(vm)
            if (time_vec(k) - time_vec(k-1)) > maxGap
                sCount = 0;
            else
                sCount = sCount + 1;
            end
            sampleValid(k) = sCount >= windowSize;  % windowSize in free-living optimizer
        end
        evalIdx = find(sampleValid)';
        if isempty(evalIdx), continue; end

        y_true_eval = y_true(evalIdx);
        y_pred_eval = y_pred(evalIdx);

        total_tp = total_tp + sum(y_true_eval == 1 & y_pred_eval == 1);
        total_fp = total_fp + sum(y_true_eval == 0 & y_pred_eval == 1);
        total_fn = total_fn + sum(y_true_eval == 1 & y_pred_eval == 0);
    end

    % Compute global F1 from pooled counts
    precision = total_tp / (total_tp + total_fp);
    recall    = total_tp / (total_tp + total_fn);

    if (precision + recall) == 0 || isnan(precision) || isnan(recall)
        globalF1 = 0;
    else
        globalF1 = 2 * (precision * recall) / (precision + recall);
    end
end

%% --- HELPER: DATA PRE-LOADER ---
function dataset = pre_load_data(files, dataPath)
    fs = 50;
    dataset = cell(length(files), 1);
    for i = 1:length(files)
        try
            fullPath = fullfile(dataPath, files(i).name);

            % Use opts to specify datetime format unambiguously
            opts = detectImportOptions(fullPath, 'Delimiter', ',');
            opts.VariableNamingRule = 'preserve';
            timeVarName = 'time';
            opts = setvaropts(opts, timeVarName, 'InputFormat', 'MM/dd/yyyy HH:mm:ss.SSS');
            temp = readtable(fullPath, opts);

            % Parse timestamps
            tRaw = temp{:, 'time'};
            if iscell(tRaw), tRaw = string(tRaw); end
            fullDateTime = datetime(tRaw, 'InputFormat', 'MM/dd/yyyy HH:mm:ss.SSS', 'Locale', 'en_US');

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
            fullDateTime = fullDateTime(keepMask);
            temp         = temp(keepMask, :);  % <-- was incorrectly 'data'

            % --- STEP 1: FIX TIME TRAVELERS (>100 day jumps) ---
            time_diffs = diff(fullDateTime);
            jumpIdx = find(abs(time_diffs) > days(100));
            for j = 1:length(jumpIdx)
                idx = jumpIdx(j);
                false_gap = time_diffs(idx) - seconds(1/fs);
                fullDateTime(idx+1:end) = fullDateTime(idx+1:end) - false_gap;
                time_diffs = diff(fullDateTime);
            end

            % --- STEP 2: GLOBAL SORT ---
            [fullDateTime, sortIdx] = sort(fullDateTime);
            temp = temp(sortIdx, :);

            % --- STEP 3: REMOVE EXACT DUPLICATE TIMESTAMPS ---
            [fullDateTime, uniqueIdx] = unique(fullDateTime);
            temp = temp(uniqueIdx, :);

            % --- STEP 4: BUILD SIGNALS ---
            s.time_vec = seconds(fullDateTime - fullDateTime(1));
            s.vm       = sqrt(temp.ax.^2 + temp.ay.^2 + temp.az.^2);

            % --- STEP 5: EXTRACT LABELS ---
            labelIdx = find(strcmpi(temp.Properties.VariableNames, 'Label'), 1);
            if ~isempty(labelIdx)
                s.y_true = temp{:, labelIdx};
                if iscell(s.y_true) || isstring(s.y_true)
                    s.y_true = str2double(s.y_true);
                end
                s.y_true(isnan(s.y_true)) = 0;
            else
                s.y_true = zeros(height(temp), 1);
            end

            % --- STEP 6: LENGTH SANITY CHECK ---
            minLen     = min([length(s.vm), length(s.y_true), length(s.time_vec)]);
            s.vm       = s.vm(1:minLen);
            s.y_true   = s.y_true(1:minLen);
            s.time_vec = s.time_vec(1:minLen);

            dataset{i} = s;

        catch ME
            fprintf('  WARNING: Skipping %s\n  Reason: %s\n', files(i).name, ME.message);
            dataset{i} = [];
        end
    end
end