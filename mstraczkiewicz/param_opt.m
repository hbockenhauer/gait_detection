%% WearGait-PD: Master Optimization (Frequency + Power + Amplitude)
clear; clc; close all;

% --- 1. CONFIGURATION ---
warning('off', 'MATLAB:table:ModifiedAndSavedVarnames'); 
dataPath = 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\WearGait-Ctrl';

% --- 2. FILE INITIALIZATION ---
files = [dir(fullfile(dataPath, 'W*.csv')); dir(fullfile(dataPath, 'N*.csv'))];
if isempty(files)
    error('No CSV files found. Check your dataPath.');
end

allData = {}; 
nFiles = length(files);
fprintf('--- PHASE 1: ROBUST DATA PRE-LOADING & AMPLITUDE CALC (%d files) ---\n', nFiles);

for i = 1:nFiles
    try
        fprintf('Processing [%2d/%d]: %-25s ', i, nFiles, files(i).name);
        data = readtable(fullfile(dataPath, files(i).name));
        cols = data.Properties.VariableNames;
        
        % 1.1 Time Conversion
        rawTime = data.Time;
        if iscell(rawTime), time = str2double(strrep(rawTime, 'sec', ''));
        elseif isduration(rawTime), time = seconds(rawTime);
        else, time = double(rawTime); end
        
        % 1.2 Column Indexing
        idxX = find(contains(cols, 'Acc') & endsWith(cols, 'X'), 1);
        idxY = find(contains(cols, 'Acc') & endsWith(cols, 'Y'), 1);
        idxZ = find(contains(cols, 'Acc') & endsWith(cols, 'Z'), 1);
        accX = double(data{:, idxX}); accY = double(data{:, idxY}); accZ = double(data{:, idxZ});
        
        % 1.3 Signal Magnitude
        vm = sqrt(accX.^2 + accY.^2 + accZ.^2);
        valid = ~isnan(vm) & ~isnan(time);
        vm = vm(valid); time = time(valid);
        fs = 1 / median(diff(time)); 
        
        % 1.4 Spectrogram & Amplitude (Std Dev) Calculation
        fs_int = round(fs);
        % We calculate Std Dev of VM in windows matching the spectrogram
        [S, F, T] = spectrogram(detrend(vm), 2*fs_int, fs_int, 512, fs);
        Cabs = abs(S).^2;
        
        % Calculate Time-Domain Amplitude (Intensity) for each window
        winSamples = 2 * fs_int;
        stepSamples = fs_int;
        numWindows = size(Cabs, 2);
        ampValues = zeros(numWindows, 1);
        for w = 1:numWindows
            startIdx = (w-1)*stepSamples + 1;
            endIdx = min(startIdx + winSamples - 1, length(vm));
            ampValues(w) = std(vm(startIdx:endIdx)); % Standard Deviation as intensity
        end
        
        % 1.5 Ground Truth Binning
        actIdx = find(contains(cols, 'Event') | contains(cols, 'Activity') | contains(cols, 'Label'), 1);
        labelData = string(data{valid, actIdx});
        y_true = contains(lower(labelData), ["walk", "stair", "gait", "jog", "free"]);
        y_true_binned = interp1(1:length(y_true), double(y_true), T*fs, 'nearest') > 0.5;
        
        allData{end+1} = struct('Cabs', Cabs, 'F', F, 'y_true', y_true_binned, 'amp', ampValues);
        fprintf('Done. (fs=%.1fHz)\n', fs);
    catch ME
        fprintf('FAILED: %s\n', ME.message);
    end
end

%% --- 3. EXPANDED GRID SEARCH OPTIMIZATION ---
fprintf('\n--- PHASE 2: EXPANDED GRID SEARCHING ---\n');

% 1. Widen and refine the search ranges
minFreqs = 0.5:0.01:1.0;       % Focus on slow gait (shuffling)
maxFreqs = 2.5:0.01:4.0;       % Capture festination/fast steps
pThreshs = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]; % Logarithmic-ish steps
aThreshs = 0.3:0.01:0.7;    % Test much higher physical intensity

[MF, XF, PT, AT] = ndgrid(minFreqs, maxFreqs, pThreshs, aThreshs);
totalCombos = numel(MF);
optResults = zeros(totalCombos, 7); 

fprintf('Testing %d combinations across 33 subjects...\n', totalCombos);

for k = 1:totalCombos
    fMin = MF(k); fMax = XF(k); pThr = PT(k); aThr = AT(k);
    TP = 0; FP = 0; FN = 0;
    
    for i = 1:length(allData)
        [maxPks, maxIdxs] = max(allData{i}.Cabs);
        peakFreqs = allData{i}.F(maxIdxs);
        amps = allData{i}.amp;
        
        % Triple Filter Logic
        y_pred = (peakFreqs(:) >= fMin & peakFreqs(:) <= fMax & ...
                  maxPks(:) > pThr & amps(:) > aThr);
              
        y_true = allData{i}.y_true(:); 
        len = min(length(y_pred), length(y_true));
        y_pred = y_pred(1:len); y_true = y_true(1:len);
        
        TP = TP + sum(y_true == 1 & y_pred == 1);
        FP = FP + sum(y_true == 0 & y_pred == 1);
        FN = FN + sum(y_true == 1 & y_pred == 0);
    end
    
    % Metrics
    prec = TP / (TP + FP); if isnan(prec), prec = 0; end
    rec  = TP / (TP + FN); if isnan(rec), rec = 0; end
    f1   = 2 * (prec * rec) / (prec + rec); if isnan(f1), f1 = 0; end
    
    optResults(k, :) = [fMin, fMax, pThr, aThr, f1, prec, rec];
    
    if mod(k, 500) == 0, fprintf('Tested %d/%d combos...\n', k, totalCombos); end
end

%% --- 4. ANALYZE TOP RESULTS ---
% Convert to table for easy viewing
resTable = array2table(optResults, 'VariableNames', ...
    {'fMin', 'fMax', 'pThresh', 'aThresh', 'F1', 'Precision', 'Recall'});
resTable = sortrows(resTable, 'F1', 'descend');

fprintf('\n--- TOP 5 PARAMETER COMBINATIONS ---\n');
disp(head(resTable, 5));

% Final Best
best = resTable(1,:);
fprintf('\nWINNER: fMin=%.2f, fMax=%.2f, pThr=%.4f, aThr=%.4f | F1=%.4f\n', ...
    best.fMin, best.fMax, best.pThresh, best.aThresh, best.F1);

% Plotting the F1 response to Amplitude Threshold
figure('Color', 'w');
scatter(optResults(:,4), optResults(:,5), 40, optResults(:,1), 'filled');
xlabel('Amplitude Threshold (g)'); ylabel('F1-Score');
title('Effect of Amplitude Filtering on Detection Performance');
cb = colorbar; ylabel(cb, 'fMin (Hz)'); grid on;