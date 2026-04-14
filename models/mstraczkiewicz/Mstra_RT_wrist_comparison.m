%% --- SigPro Real-Time Wrist Comparison Evaluator ---
% Compare affected wrist, unaffected wrist, and fusion strategies for QSense Clinic patients
% Based on Mstra_RT.m with wrist-specific evaluation

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

% Start from script directory and go up to find project root
projectRoot = scriptDir;
for k = 1:10
    hasModels = exist(fullfile(projectRoot, 'models'), 'dir') == 7;
    hasDatasets = exist(fullfile(projectRoot, 'Datasets'), 'dir') == 7;
    
    if hasModels && hasDatasets
        break;
    end
    
    parentDir = fileparts(projectRoot);
    if strcmp(parentDir, projectRoot)
        % Reached filesystem root without finding project
        break;
    end
    projectRoot = parentDir;
end

outputsRoot = fullfile(projectRoot, 'outputs');
resultsDir = fullfile(outputsRoot, 'results');
plotsRoot = fullfile(outputsRoot, 'plots');

% Data path search (QSense Clinic can be in different locations)
qSenseCandidates = {
    fullfile(projectRoot, 'Clinical')
    fullfile(projectRoot, 'Datasets', 'Clinical')
    fullfile(projectRoot, 'QSense_clinic')
    fullfile(projectRoot, 'Datasets', 'QSense_clinic')
};

dataPaths = qSenseCandidates(cellfun(@(p) exist(p, 'dir') == 7, qSenseCandidates));
dataPaths = unique(dataPaths, 'stable');

if isempty(dataPaths)
    fprintf('projectRoot: %s\n', projectRoot);
    fprintf('Checked paths:\n');
    for i = 1:length(qSenseCandidates)
        fprintf('  %s - exists: %d\n', qSenseCandidates{i}, exist(qSenseCandidates{i}, 'dir'));
    end
    error('No QSense clinic dataset directory found. projectRoot=%s', projectRoot);
end

% SigPro thresholds (from Mstra_RT.m)
F_MIN = 0.097033; F_MAX = 7.1325;
P_MIN = 2.3599; P_MAX = 496.76;
A_MIN = 120.77; A_MAX = 1105.1;

fs = 50;
windowSize = 2 * fs;
stepSize = 1 * fs;

% Affected side per patient in QSense Clinic [sub1, sub2, sub3...]
% "RW" = Right Wrist (affected), "LW" = Left Wrist (affected)
affected_wrist_mapping = containers.Map({'sub1', 'sub2', 'sub3'}, {'RW', 'LW', 'LW'});

if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% --- 2. INITIALIZE SUMMARIES ---
summaryResults = table();

% --- 3. MAIN LOOP ---
for d = 1:length(dataPaths)
    dataPath = dataPaths{d};
    if ~exist(dataPath, 'dir'), continue; end
    
    [~, datasetName] = fileparts(dataPath);
    
    subDirs = dir(dataPath);
    subDirs = subDirs([subDirs.isdir] & ~ismember({subDirs.name}, {'.', '..'}));
    
    fprintf('\n======================================================================\n');
    fprintf('WRIST COMPARISON ANALYSIS: %s\n', datasetName);
    fprintf('======================================================================\n\n');
    
    for i = 1:length(subDirs)
        folderName = subDirs(i).name;
        folderPath = fullfile(dataPath, folderName);
        
        % Extract subject number (e.g., "sub1" from "sub1_someinfo")
        subjectStr = strsplit(folderName, '_');
        subjectNum = subjectStr{1};
        
        % Get affected wrist for this subject
        if isKey(affected_wrist_mapping, subjectNum)
            affectedWrist = affected_wrist_mapping(subjectNum);
        else
            affectedWrist = 'unknown';
        end
        
        if strcmp(affectedWrist, 'RW')
            unaffectedWrist = 'LW';
        else
            unaffectedWrist = 'RW';
        end
        
        fprintf('\n--- %s (Affected: %s, Unaffected: %s) ---\n', folderName, affectedWrist, unaffectedWrist);
        
        % Evaluate individual wrists
        rw_result = evaluate_wrist(folderPath, 'RW', fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        lw_result = evaluate_wrist(folderPath, 'LW', fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        
        % Evaluate fusion strategies
        fusion_acc_result = evaluate_fusion_acc(folderPath, fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        fusion_voting_or_result = evaluate_fusion_voting_or(folderPath, fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        fusion_voting_and_result = evaluate_fusion_voting_and(folderPath, fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        
        % Right Wrist (determine if affected or unaffected)
        if ~isempty(rw_result)
            if isequal(affectedWrist, 'RW')
                rw_type = 'affected';
            else
                rw_type = 'unaffected';
            end
            fprintf('  Evaluating RIGHT wrist (%s)...\n', rw_type);
            fprintf('    Precision: %.3f | Recall: %.3f | F1: %.3f | Accuracy: %.3f\n', ...
                rw_result.precision, rw_result.recall, rw_result.f1, rw_result.accuracy);
            
            summaryResults = [summaryResults; table({subjectNum}, {subjectNum}, {rw_type}, {'right'}, ...
                rw_result.precision, rw_result.recall, rw_result.f1, rw_result.accuracy, ...
                'VariableNames', {'Subject', 'Subject_Num', 'Condition', 'Wrist_Name', 'Precision', 'Recall', 'F1', 'Accuracy'})];
        else
            fprintf('  Evaluating RIGHT wrist...\n');
            fprintf('    Could not evaluate right wrist\n');
        end
        
        % Left Wrist (determine if affected or unaffected)
        if ~isempty(lw_result)
            if isequal(affectedWrist, 'LW')
                lw_type = 'affected';
            else
                lw_type = 'unaffected';
            end
            fprintf('  Evaluating LEFT wrist (%s)...\n', lw_type);
            fprintf('    Precision: %.3f | Recall: %.3f | F1: %.3f | Accuracy: %.3f\n', ...
                lw_result.precision, lw_result.recall, lw_result.f1, lw_result.accuracy);
            
            summaryResults = [summaryResults; table({subjectNum}, {subjectNum}, {lw_type}, {'left'}, ...
                lw_result.precision, lw_result.recall, lw_result.f1, lw_result.accuracy, ...
                'VariableNames', {'Subject', 'Subject_Num', 'Condition', 'Wrist_Name', 'Precision', 'Recall', 'F1', 'Accuracy'})];
        else
            fprintf('  Evaluating LEFT wrist...\n');
            fprintf('    Could not evaluate left wrist\n');
        end
        
        % Fusion strategies
        if ~isempty(fusion_acc_result)
            fprintf('  Evaluating BOTH wrists (ACC averaged)...\n');
            fprintf('    Precision: %.3f | Recall: %.3f | F1: %.3f | Accuracy: %.3f\n', ...
                fusion_acc_result.precision, fusion_acc_result.recall, fusion_acc_result.f1, fusion_acc_result.accuracy);
            
            summaryResults = [summaryResults; table({subjectNum}, {subjectNum}, {'fusion'}, {'both_acc_avg'}, ...
                fusion_acc_result.precision, fusion_acc_result.recall, fusion_acc_result.f1, fusion_acc_result.accuracy, ...
                'VariableNames', {'Subject', 'Subject_Num', 'Condition', 'Wrist_Name', 'Precision', 'Recall', 'F1', 'Accuracy'})];
        else
            fprintf('  Evaluating BOTH wrists (ACC averaged)...\n');
            fprintf('    Could not evaluate both wrists (ACC averaged)\n');
        end
        
        if ~isempty(fusion_voting_or_result)
            fprintf('  Evaluating BOTH wrists (Voting OR)...\n');
            fprintf('    Precision: %.3f | Recall: %.3f | F1: %.3f | Accuracy: %.3f\n', ...
                fusion_voting_or_result.precision, fusion_voting_or_result.recall, fusion_voting_or_result.f1, fusion_voting_or_result.accuracy);
            
            summaryResults = [summaryResults; table({subjectNum}, {subjectNum}, {'fusion'}, {'voting_or'}, ...
                fusion_voting_or_result.precision, fusion_voting_or_result.recall, fusion_voting_or_result.f1, fusion_voting_or_result.accuracy, ...
                'VariableNames', {'Subject', 'Subject_Num', 'Condition', 'Wrist_Name', 'Precision', 'Recall', 'F1', 'Accuracy'})];
        else
            fprintf('  Evaluating BOTH wrists (Voting OR)...\n');
            fprintf('    Could not evaluate both wrists (Voting OR)\n');
        end
        
        if ~isempty(fusion_voting_and_result)
            fprintf('  Evaluating BOTH wrists (Voting AND)...\n');
            fprintf('    Precision: %.3f | Recall: %.3f | F1: %.3f | Accuracy: %.3f\n', ...
                fusion_voting_and_result.precision, fusion_voting_and_result.recall, fusion_voting_and_result.f1, fusion_voting_and_result.accuracy);
            
            summaryResults = [summaryResults; table({subjectNum}, {subjectNum}, {'fusion'}, {'voting_and'}, ...
                fusion_voting_and_result.precision, fusion_voting_and_result.recall, fusion_voting_and_result.f1, fusion_voting_and_result.accuracy, ...
                'VariableNames', {'Subject', 'Subject_Num', 'Condition', 'Wrist_Name', 'Precision', 'Recall', 'F1', 'Accuracy'})];
        else
            fprintf('  Evaluating BOTH wrists (Voting AND)...\n');
            fprintf('    Could not evaluate both wrists (Voting AND)\n');
        end
    end
end

% Save results
outputFile = fullfile(resultsDir, 'SigPro_wrist_comparison_results.csv');
writetable(summaryResults, outputFile);
fprintf('\n======================================================================\n');
fprintf('Results saved to: %s\n', outputFile);
fprintf('======================================================================\n\n');

%% --- HELPER FUNCTIONS ---

function result = evaluate_wrist(folderPath, wristLabel, fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX)
    % Evaluate a single wrist
    
    % Get file path based on wrist label
    if strcmp(wristLabel, 'RW')
        fileName = 's1_1RW.txt';
    elseif strcmp(wristLabel, 'LW')
        fileName = 's2_2LW.txt';
    else
        result = [];
        return;
    end
    
    fullFilePath = fullfile(folderPath, fileName);
    if ~isfile(fullFilePath)
        result = [];
        return;
    end
    
    try
        % Load data (now returns energy)
        [time_vec, vm_all, y_true, energy] = load_and_clean_qsense_file(fullFilePath, fs);
        
        if isempty(time_vec) || length(time_vec) < windowSize
            result = [];
            return;
        end
        
        % Run real-time simulation
        [y_pred_rt, sampleValid, ~, ~, ~, ~] = run_rt_detection(time_vec, vm_all, energy, fs, windowSize, stepSize, ...
            F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        
        % Calculate metrics using valid samples
        [prec, rec, f1, acc] = calculate_metrics(y_true, y_pred_rt, sampleValid);
        
        result = struct('precision', prec, 'recall', rec, 'f1', f1, 'accuracy', acc);
    catch ME
        fprintf('      Error: %s\n', ME.message);
        result = [];
    end
end

function result = evaluate_fusion_acc(folderPath, fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX)
    % Evaluate fusion by averaging accelerations and energies
    
    rw_file = fullfile(folderPath, 's1_1RW.txt');
    lw_file = fullfile(folderPath, 's2_2LW.txt');
    
    rw_exists = isfile(rw_file);
    lw_exists = isfile(lw_file);
    
    % If neither file exists, return empty
    if ~rw_exists && ~lw_exists
        result = [];
        return;
    end
    
    % If only one wrist is available, fall back to single wrist evaluation
    if ~rw_exists
        result = evaluate_wrist(folderPath, 'LW', fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        return;
    end
    
    if ~lw_exists
        result = evaluate_wrist(folderPath, 'RW', fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        return;
    end
    
    try
        [time_rw, vm_rw, y_true_rw, energy_rw] = load_and_clean_qsense_file(rw_file, fs);
        [time_lw, vm_lw, y_true_lw, energy_lw] = load_and_clean_qsense_file(lw_file, fs);
        
        % Align times and average accelerations + energies
        [time_fused, vm_fused, y_true_fused, energy_fused] = align_and_average_vms(time_rw, vm_rw, y_true_rw, energy_rw, time_lw, vm_lw, y_true_lw, energy_lw);
        
        if isempty(time_fused) || length(time_fused) < windowSize
            result = [];
            return;
        end
        
        % Run real-time simulation on fused data
        [y_pred_rt, sampleValid, ~, ~, ~, ~] = run_rt_detection(time_fused, vm_fused, energy_fused, fs, windowSize, stepSize, ...
            F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        
        % Calculate metrics
        [prec, rec, f1, acc] = calculate_metrics(y_true_fused, y_pred_rt, sampleValid);
        
        result = struct('precision', prec, 'recall', rec, 'f1', f1, 'accuracy', acc);
    catch ME
        fprintf('      Error: %s\n', ME.message);
        result = [];
    end
end

function result = evaluate_fusion_voting_or(folderPath, fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX)
    % Voting OR: Predict gait if at least ONE wrist says gait
    % Run detector on each wrist separately, then OR the predictions
    
    rw_file = fullfile(folderPath, 's1_1RW.txt');
    lw_file = fullfile(folderPath, 's2_2LW.txt');
    
    rw_exists = isfile(rw_file);
    lw_exists = isfile(lw_file);
    
    % If neither file exists, return empty
    if ~rw_exists && ~lw_exists
        result = [];
        return;
    end
    
    % If only one wrist is available, fall back to single wrist evaluation
    if ~rw_exists
        result = evaluate_wrist(folderPath, 'LW', fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        return;
    end
    
    if ~lw_exists
        result = evaluate_wrist(folderPath, 'RW', fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        return;
    end
    
    try
        [time_rw, vm_rw, y_true_rw, energy_rw] = load_and_clean_qsense_file(rw_file, fs);
        [time_lw, vm_lw, y_true_lw, energy_lw] = load_and_clean_qsense_file(lw_file, fs);
        
        % Get predictions from each wrist
        [y_pred_rw, sampleValid_rw, ~, ~, ~, ~] = run_rt_detection(time_rw, vm_rw, energy_rw, fs, windowSize, stepSize, ...
            F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        [y_pred_lw, sampleValid_lw, ~, ~, ~, ~] = run_rt_detection(time_lw, vm_lw, energy_lw, fs, windowSize, stepSize, ...
            F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        
        % Align predictions by time and apply OR voting
        [y_pred_voting, sampleValid_voting, y_true_voting] = align_and_voting_or(time_rw, y_pred_rw, sampleValid_rw, y_true_rw, ...
            time_lw, y_pred_lw, sampleValid_lw, y_true_lw);
        
        if isempty(y_pred_voting)
            result = [];
            return;
        end
        
        % Calculate metrics
        [prec, rec, f1, acc] = calculate_metrics(y_true_voting, y_pred_voting, sampleValid_voting);
        
        result = struct('precision', prec, 'recall', rec, 'f1', f1, 'accuracy', acc);
    catch ME
        fprintf('      Error: %s\n', ME.message);
        result = [];
    end
end

function result = evaluate_fusion_voting_and(folderPath, fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX)
    % Voting AND: Predict gait only if BOTH wrists say gait
    % Run detector on each wrist separately, then AND the predictions
    
    rw_file = fullfile(folderPath, 's1_1RW.txt');
    lw_file = fullfile(folderPath, 's2_2LW.txt');
    
    rw_exists = isfile(rw_file);
    lw_exists = isfile(lw_file);
    
    % If neither file exists, return empty
    if ~rw_exists && ~lw_exists
        result = [];
        return;
    end
    
    % If only one wrist is available, fall back to single wrist evaluation
    if ~rw_exists
        result = evaluate_wrist(folderPath, 'LW', fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        return;
    end
    
    if ~lw_exists
        result = evaluate_wrist(folderPath, 'RW', fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        return;
    end
    
    try
        [time_rw, vm_rw, y_true_rw, energy_rw] = load_and_clean_qsense_file(rw_file, fs);
        [time_lw, vm_lw, y_true_lw, energy_lw] = load_and_clean_qsense_file(lw_file, fs);
        
        % Get predictions from each wrist
        [y_pred_rw, sampleValid_rw, ~, ~, ~, ~] = run_rt_detection(time_rw, vm_rw, energy_rw, fs, windowSize, stepSize, ...
            F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        [y_pred_lw, sampleValid_lw, ~, ~, ~, ~] = run_rt_detection(time_lw, vm_lw, energy_lw, fs, windowSize, stepSize, ...
            F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX);
        
        % Align predictions by time and apply AND voting
        [y_pred_voting, sampleValid_voting, y_true_voting] = align_and_voting_and(time_rw, y_pred_rw, sampleValid_rw, y_true_rw, ...
            time_lw, y_pred_lw, sampleValid_lw, y_true_lw);
        
        if isempty(y_pred_voting)
            result = [];
            return;
        end
        
        % Calculate metrics
        [prec, rec, f1, acc] = calculate_metrics(y_true_voting, y_pred_voting, sampleValid_voting);
        
        result = struct('precision', prec, 'recall', rec, 'f1', f1, 'accuracy', acc);
    catch ME
        fprintf('      Error: %s\n', ME.message);
        result = [];
    end
end

function [time_vec, vm_all, y_true, energy] = load_and_clean_qsense_file(filePath, fs)
    % Load and clean QSense CSV file
    
    opts = detectImportOptions(filePath);
    opts.VariableNamingRule = 'preserve';
    opts = setvartype(opts, [1, 2], 'string');
    data = readtable(filePath, opts);
    
    dateTimeStr = string(data{:,1}) + " " + string(data{:,2});
    fullDateTime = datetime(dateTimeStr, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
    
    % Remove backwards time jumps
    runningMax = fullDateTime(1);
    keepMask = true(length(fullDateTime), 1);
    for k = 1:length(fullDateTime)
        if fullDateTime(k) < runningMax
            keepMask(k) = false;
        else
            runningMax = fullDateTime(k);
        end
    end
    
    fullDateTime = fullDateTime(keepMask);
    data = data(keepMask, :);
    
    % Fix time travelers (100+ day jumps)
    time_diffs = diff(fullDateTime);
    jumpIdx = find(abs(time_diffs) > days(100));
    
    for j = 1:length(jumpIdx)
        idx = jumpIdx(j);
        false_gap = time_diffs(idx) - seconds(1/fs);
        fullDateTime(idx+1:end) = fullDateTime(idx+1:end) - false_gap;
        time_diffs = diff(fullDateTime);
    end
    
    % Sort and remove duplicates
    [fullDateTime, sortIdx] = sort(fullDateTime);
    data = data(sortIdx, :);
    
    [fullDateTime, uniqueIdx] = unique(fullDateTime);
    data = data(uniqueIdx, :);
    
    % Create time vector
    time_vec = seconds(fullDateTime - fullDateTime(1));
    
    % Extract accelerations and energy
    vm_all = sqrt(data{:,6}.^2 + data{:,7}.^2 + data{:,8}.^2);
    
    % Try to extract energy from column 13
    try
        energy = data{:,13};
    catch
        % If column 13 doesn't exist, compute energy as std of acceleration
        energy = zeros(length(vm_all), 1);
        for idx = 1:length(vm_all)
            energy(idx) = std([data{idx,6}, data{idx,7}, data{idx,8}]);
        end
    end
    
    % Extract labels
    varNames = data.Properties.VariableNames;
    labelIdx = find(strcmpi(varNames, 'Label'), 1);
    
    if ~isempty(labelIdx)
        raw_gt = data{:, labelIdx};
        if iscell(raw_gt) || isstring(raw_gt)
            raw_gt = str2double(raw_gt);
        end
        raw_gt(isnan(raw_gt)) = 0;
        y_true = double(raw_gt);
    else
        y_true = zeros(length(vm_all), 1);
    end
    
    if length(y_true) ~= length(vm_all)
        y_true = y_true(1:length(vm_all));
    end
    
    if length(energy) ~= length(vm_all)
        energy = energy(1:length(vm_all));
    end
end

function [time_fused, vm_fused, y_true_fused, energy_fused] = align_and_average_vms(time_rw, vm_rw, y_true_rw, energy_rw, time_lw, vm_lw, y_true_lw, energy_lw)
    % Align two wrists by time and average VMs and energies
    time_tol = 0.02;
    i = 1; j = 1;
    
    time_fused = [];
    vm_fused = [];
    y_true_fused = [];
    energy_fused = [];
    
    while i <= length(time_rw) && j <= length(time_lw)
        dt = time_rw(i) - time_lw(j);
        
        if abs(dt) <= time_tol
            time_fused = [time_fused; (time_rw(i) + time_lw(j)) / 2];
            vm_fused = [vm_fused; (vm_rw(i) + vm_lw(j)) / 2];
            y_true_fused = [y_true_fused; max(y_true_rw(i), y_true_lw(j))];
            energy_fused = [energy_fused; (energy_rw(i) + energy_lw(j)) / 2];
            i = i + 1;
            j = j + 1;
        elseif dt < 0
            time_fused = [time_fused; time_rw(i)];
            vm_fused = [vm_fused; vm_rw(i)];
            y_true_fused = [y_true_fused; y_true_rw(i)];
            energy_fused = [energy_fused; energy_rw(i)];
            i = i + 1;
        else
            time_fused = [time_fused; time_lw(j)];
            vm_fused = [vm_fused; vm_lw(j)];
            y_true_fused = [y_true_fused; y_true_lw(j)];
            energy_fused = [energy_fused; energy_lw(j)];
            j = j + 1;
        end
    end
    
    while i <= length(time_rw)
        time_fused = [time_fused; time_rw(i)];
        vm_fused = [vm_fused; vm_rw(i)];
        y_true_fused = [y_true_fused; y_true_rw(i)];
        energy_fused = [energy_fused; energy_rw(i)];
        i = i + 1;
    end
    
    while j <= length(time_lw)
        time_fused = [time_fused; time_lw(j)];
        vm_fused = [vm_fused; vm_lw(j)];
        y_true_fused = [y_true_fused; y_true_lw(j)];
        energy_fused = [energy_fused; energy_lw(j)];
        j = j + 1;
    end
end

function [time_fused, vm_fused, y_true_fused, energy_fused] = align_and_max_vms(time_rw, vm_rw, y_true_rw, energy_rw, time_lw, vm_lw, y_true_lw, energy_lw)
    % Align two wrists by time and take element-wise max of VMs and max of energies
    time_tol = 0.02;
    i = 1; j = 1;
    
    time_fused = [];
    vm_fused = [];
    y_true_fused = [];
    energy_fused = [];
    
    while i <= length(time_rw) && j <= length(time_lw)
        dt = time_rw(i) - time_lw(j);
        
        if abs(dt) <= time_tol
            time_fused = [time_fused; (time_rw(i) + time_lw(j)) / 2];
            vm_fused = [vm_fused; max(vm_rw(i), vm_lw(j))];
            y_true_fused = [y_true_fused; max(y_true_rw(i), y_true_lw(j))];
            energy_fused = [energy_fused; max(energy_rw(i), energy_lw(j))];
            i = i + 1;
            j = j + 1;
        elseif dt < 0
            time_fused = [time_fused; time_rw(i)];
            vm_fused = [vm_fused; vm_rw(i)];
            y_true_fused = [y_true_fused; y_true_rw(i)];
            energy_fused = [energy_fused; energy_rw(i)];
            i = i + 1;
        else
            time_fused = [time_fused; time_lw(j)];
            vm_fused = [vm_fused; vm_lw(j)];
            y_true_fused = [y_true_fused; y_true_lw(j)];
            energy_fused = [energy_fused; energy_lw(j)];
            j = j + 1;
        end
    end
    
    while i <= length(time_rw)
        time_fused = [time_fused; time_rw(i)];
        vm_fused = [vm_fused; vm_rw(i)];
        y_true_fused = [y_true_fused; y_true_rw(i)];
        energy_fused = [energy_fused; energy_rw(i)];
        i = i + 1;
    end
    
    while j <= length(time_lw)
        time_fused = [time_fused; time_lw(j)];
        vm_fused = [vm_fused; vm_lw(j)];
        y_true_fused = [y_true_fused; y_true_lw(j)];
        energy_fused = [energy_fused; energy_lw(j)];
        j = j + 1;
    end
end

function [time_fused, vm_fused, y_true_fused, energy_fused] = align_and_min_vms(time_rw, vm_rw, y_true_rw, energy_rw, time_lw, vm_lw, y_true_lw, energy_lw)
    % Align two wrists by time and take element-wise min of VMs and min of energies
    time_tol = 0.02;
    i = 1; j = 1;
    
    time_fused = [];
    vm_fused = [];
    y_true_fused = [];
    energy_fused = [];
    
    while i <= length(time_rw) && j <= length(time_lw)
        dt = time_rw(i) - time_lw(j);
        
        if abs(dt) <= time_tol
            time_fused = [time_fused; (time_rw(i) + time_lw(j)) / 2];
            vm_fused = [vm_fused; min(vm_rw(i), vm_lw(j))];
            y_true_fused = [y_true_fused; max(y_true_rw(i), y_true_lw(j))];
            energy_fused = [energy_fused; min(energy_rw(i), energy_lw(j))];
            i = i + 1;
            j = j + 1;
        elseif dt < 0
            time_fused = [time_fused; time_rw(i)];
            vm_fused = [vm_fused; vm_rw(i)];
            y_true_fused = [y_true_fused; y_true_rw(i)];
            energy_fused = [energy_fused; energy_rw(i)];
            i = i + 1;
        else
            time_fused = [time_fused; time_lw(j)];
            vm_fused = [vm_fused; vm_lw(j)];
            y_true_fused = [y_true_fused; y_true_lw(j)];
            energy_fused = [energy_fused; energy_lw(j)];
            j = j + 1;
        end
    end
    
    while i <= length(time_rw)
        time_fused = [time_fused; time_rw(i)];
        vm_fused = [vm_fused; vm_rw(i)];
        y_true_fused = [y_true_fused; y_true_rw(i)];
        energy_fused = [energy_fused; energy_rw(i)];
        i = i + 1;
    end
    
    while j <= length(time_lw)
        time_fused = [time_fused; time_lw(j)];
        vm_fused = [vm_fused; vm_lw(j)];
        y_true_fused = [y_true_fused; y_true_lw(j)];
        energy_fused = [energy_fused; energy_lw(j)];
        j = j + 1;
    end
end

function [y_pred_voted, sampleValid_voted, y_true_voted] = align_and_voting_or(time_rw, y_pred_rw, sampleValid_rw, y_true_rw, time_lw, y_pred_lw, sampleValid_lw, y_true_lw)
    % Align two wrist predictions by time and apply OR voting (if either wrist says gait, output 1)
    time_tol = 0.02;
    i = 1; j = 1;
    
    y_pred_voted = [];
    sampleValid_voted = [];
    y_true_voted = [];
    
    while i <= length(time_rw) && j <= length(time_lw)
        dt = time_rw(i) - time_lw(j);
        
        if abs(dt) <= time_tol
            % Both wrists have aligned samples
            y_pred_voted = [y_pred_voted; y_pred_rw(i) | y_pred_lw(j)];  % OR voting
            sampleValid_voted = [sampleValid_voted; sampleValid_rw(i) & sampleValid_lw(j)];  % Both must be valid
            y_true_voted = [y_true_voted; max(y_true_rw(i), y_true_lw(j))];  % Ground truth is OR of both
            i = i + 1;
            j = j + 1;
        elseif dt < 0
            % RW sample is earlier, skip it
            i = i + 1;
        else
            % LW sample is earlier, skip it
            j = j + 1;
        end
    end
end

function [y_pred_voted, sampleValid_voted, y_true_voted] = align_and_voting_and(time_rw, y_pred_rw, sampleValid_rw, y_true_rw, time_lw, y_pred_lw, sampleValid_lw, y_true_lw)
    % Align two wrist predictions by time and apply AND voting (if both wrists say gait, output 1)
    time_tol = 0.02;
    i = 1; j = 1;
    
    y_pred_voted = [];
    sampleValid_voted = [];
    y_true_voted = [];
    
    while i <= length(time_rw) && j <= length(time_lw)
        dt = time_rw(i) - time_lw(j);
        
        if abs(dt) <= time_tol
            % Both wrists have aligned samples
            y_pred_voted = [y_pred_voted; y_pred_rw(i) & y_pred_lw(j)];  % AND voting
            sampleValid_voted = [sampleValid_voted; sampleValid_rw(i) & sampleValid_lw(j)];  % Both must be valid
            y_true_voted = [y_true_voted; max(y_true_rw(i), y_true_lw(j))];  % Ground truth is OR of both
            i = i + 1;
            j = j + 1;
        elseif dt < 0
            % RW sample is earlier, skip it
            i = i + 1;
        else
            % LW sample is earlier, skip it
            j = j + 1;
        end
    end
end

function [y_pred_rt, sampleValid, rt_T, rt_peakF, rt_maxPk, rt_ampVal] = run_rt_detection(time_vec, vm_all, energy, fs, windowSize, stepSize, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX)
    % Real-time detection simulation with sample validity mask
    
    totalSamples = length(vm_all);
    y_pred_rt = zeros(totalSamples, 1);
    sampleValid = false(totalSamples, 1);  % Track which samples are valid for evaluation
    
    circularBuffer = ones(windowSize, 1) * vm_all(1);
    detectionState = 0;
    samplesSinceReset = 0;
    
    rt_T = [];
    rt_peakF = [];
    rt_maxPk = [];
    rt_ampVal = [];
    
    maxGap = 1.5 / fs;
    
    for s = 2:totalSamples
        dt = time_vec(s) - time_vec(s-1);
        
        if dt > maxGap
            circularBuffer(:) = vm_all(s);
            detectionState = 0;
            samplesSinceReset = 0;
            continue;
        end
        
        samplesSinceReset = samplesSinceReset + 1;
        circularBuffer = [circularBuffer(2:end); vm_all(s)];
        
        % Mark sample as valid if we have enough history since last gap
        if samplesSinceReset >= windowSize
            sampleValid(s) = true;
        end
        
        if mod(s, stepSize) == 0 && s >= windowSize
            % Use actual energy from file (indexed at current sample)
            energyVal = energy(s);
            [isGait, newState, m] = run_mstra_fast_rt_with_metrics(circularBuffer, fs, F_MIN, F_MAX, P_MIN, P_MAX, A_MIN, A_MAX, detectionState, energyVal);
            detectionState = newState;
            y_pred_rt(s-stepSize+1:s) = double(isGait);
            
            rt_T = [rt_T, time_vec(s)];
            rt_peakF = [rt_peakF, m.peakF];
            rt_maxPk = [rt_maxPk, m.maxPk];
            rt_ampVal = [rt_ampVal, m.ampVal];
        end
    end
end

function [prec, rec, f1, acc] = calculate_metrics(y_true, y_pred, sampleValid)
    % Calculate precision, recall, F1, and accuracy using only valid samples
    
    if nargin < 3
        sampleValid = true(length(y_true), 1);  % Default: all samples valid
    end
    
    evalIdx = find(sampleValid);
    if isempty(evalIdx)
        prec = 0; rec = 0; f1 = 0; acc = 0;
        return;
    end
    
    tp = sum(y_true(evalIdx) == 1 & y_pred(evalIdx) == 1);
    tn = sum(y_true(evalIdx) == 0 & y_pred(evalIdx) == 0);
    fp = sum(y_true(evalIdx) == 0 & y_pred(evalIdx) == 1);
    fn = sum(y_true(evalIdx) == 1 & y_pred(evalIdx) == 0);
    
    prec = tp / (tp + fp);
    if isnan(prec), prec = 0; end
    
    rec = tp / (tp + fn);
    if isnan(rec), rec = 0; end
    
    f1 = 2 * (prec * rec) / (prec + rec);
    if isnan(f1), f1 = 0; end
    
    acc = (tp + tn) / (tp + tn + fp + fn);
    if isnan(acc), acc = 0; end
end

function [finalDecision, newState, metrics] = run_mstra_fast_rt_with_metrics(winData, fs, fMin, fMax, pMin, pMax, aMin, aMax, prevState, energy)
    % SigPro real-time gait detection with metrics
    
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
    
    % Reduce to 2 consecutive seconds (1 history element)
    newState = rawDecision;
    finalDecision = prevState & rawDecision;
end
