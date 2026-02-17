%% WearGait-PD: Dual-Wrist Global Summary
clear; clc; close all;

% --- 1. CONFIGURATION ---
dataPath = 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\WearGait-PD';

% --- 2. FILE INITIALIZATION ---
wFiles = dir(fullfile(dataPath, 'W*.csv')); 
nFiles = dir(fullfile(dataPath, 'N*.csv')); 
files = [wFiles; nFiles];

if isempty(files)
    error('No CSV files found. Check your dataPath.');
end

summaryResults = table();
fprintf('Processing %d files (Checking both wrists)...\n', length(files));
fprintf('%-22s | %-8s | %-8s | %-8s | %-8s\n', 'Subject_Wrist', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
fprintf('--------------------------------------------------------------------------------\n');

% --- 3. MAIN PROCESSING LOOP ---
for i = 1:length(files)
    fileName = files(i).name;
    subjectID = strrep(fileName, '.csv', '');
    
    try
        opts = detectImportOptions(fullfile(dataPath, fileName));
        opts.VariableNamingRule = 'preserve';
        data = readtable(fullfile(dataPath, fileName), opts);
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
            fs = round(1 / median(diff(timeClean(1:min(1000, end)))));
            [y_pred, steps] = run_straczkiewicz_lite(vm, fs);
            
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

function [wi, steps] = run_straczkiewicz_lite(vm, fs)
    fs_int = round(fs);
    nSec = floor(length(vm)/fs_int);
    [S, F, T_vec] = spectrogram(detrend(vm), 2*fs_int, fs_int, 512, fs);
    Cabs = abs(S).^2;
    wi_raw = zeros(size(T_vec));
    for i = 1:length(T_vec)
        [pks, locs] = findpeaks(Cabs(:,i), F);
        if isempty(pks), continue; end
        [maxPk, maxIdx] = max(pks);
        if locs(maxIdx) >= 0.6 && locs(maxIdx) <= 3.4 && maxPk > 0.0001 % Lowered threshold for better recall
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