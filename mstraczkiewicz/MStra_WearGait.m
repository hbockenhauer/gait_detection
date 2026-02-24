%% --- MStra detection on WearGait-PD wrist data ---

clear; clc;

% --- Straczkiewicz Parameters ---
min_amp   = 0.1;          % Minimum amplitude (g)
T         = 3;             % Minimum walking duration (s)
alpha     = 2;
beta      = 2;
step_freq = [0.5 3.5];     % Cadence range (Hz)

% --- Paths ---
dataPath = 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\WearGait-PD';
PlotPath = 'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\mstraczkiewicz\MStraPlots';

summaryResults = table();

fprintf('\nProcessing WearGait dataset: %s\n\n', dataPath);
fprintf('%-30s | %-8s | %-8s | %-8s | %-8s\n', ...
    'Subject_Wrist','Accuracy','Precision','Recall','F1');
fprintf('-------------------------------------------------------------------------------\n');

files = dir(fullfile(dataPath,'*.csv'));

for f = 1:length(files)

    fullFilePath = fullfile(dataPath, files(f).name);
    subjectName = erase(files(f).name,'.csv');

    try
        % --- Load WearGait File ---
        dataWG = load_weargait_data(fullFilePath);

        wrists = {'right','left'};

        for w = 1:2

            wristName = wrists{w};
            wristData = dataWG.(wristName);

            time   = wristData.time;
            accX   = wristData.acc_x;
            accY   = wristData.acc_y;
            accZ   = wristData.acc_z;
            labels = wristData.labels;

            if length(time) < 200
                continue;
            end

            % --- Sampling frequency ---
            fs = round(1 / median(diff(time)));

            % --- Proper Straczkiewicz delta ---
            delta = round(0.5 * fs);

            % --- Remove first 10 seconds ---
            keepIdx = time >= 10;
            accX = accX(keepIdx);
            accY = accY(keepIdx);
            accZ = accZ(keepIdx);
            labels = labels(keepIdx);

            % --- Vector magnitude ---
            vm = sqrt(accX.^2 + accY.^2 + accZ.^2);
            vm = vm(:);

            if length(vm) < fs * T
                continue;
            end

            % --- Ground truth from GeneralEvent ---
            isWalking = contains(lower(labels),"walk") | ...
                        contains(lower(labels),"stairs");

            y_true = double(isWalking);
            y_true = y_true(:);

            % --- Run Straczkiewicz detector ---
            [wi, steps, cad] = find_walking(vm, fs, min_amp, T, ...
                                            delta, alpha, beta, step_freq);

            wi = wi(:);

            % Align lengths safely
            minLen = min(length(wi), length(y_true));
            wi = wi(1:minLen);
            y_true = y_true(1:minLen);

            % --- Metrics ---
            tp = sum(y_true == 1 & wi == 1);
            tn = sum(y_true == 0 & wi == 0);
            fp = sum(y_true == 0 & wi == 1);
            fn = sum(y_true == 1 & wi == 0);

            total = tp + tn + fp + fn;
            if total == 0
                continue;
            end

            acc  = (tp + tn) / total;
            prec = tp / (tp + fp); if (tp + fp) == 0, prec = 1; end
            rec  = tp / (tp + fn); if (tp + fn) == 0, rec = 1; end
            f1   = 2 * (prec * rec) / (prec + rec); if (prec + rec) == 0, f1 = 0; end

            steps_count = length(steps);

            % --- Store results ---
            fullID = sprintf('%s_%s', subjectName, wristName);

            resRow = table({fullID}, {subjectName}, {wristName}, ...
                           acc, prec, rec, f1, steps_count, ...
                           tp, tn, fp, fn, ...
                'VariableNames', {'ID','Subject','Wrist','Accuracy',...
                                  'Precision','Recall','F1','Steps',...
                                  'TP','TN','FP','FN'});

            summaryResults = [summaryResults; resRow];

            fprintf('%-30s | %8.4f | %8.4f | %8.4f | %8.4f\n', ...
                fullID, acc, prec, rec, f1);

        end

    catch ME
        fprintf('%-30s | ERROR: %s\n', subjectName, ME.message);
    end

end

%% --- GLOBAL STATISTICS ---
if ~isempty(summaryResults)

    TP = sum(summaryResults.TP);
    TN = sum(summaryResults.TN);
    FP = sum(summaryResults.FP);
    FN = sum(summaryResults.FN);

    globalAcc  = (TP + TN) / (TP + TN + FP + FN);
    globalPrec = TP / (TP + FP); if (TP + FP) == 0, globalPrec = 1; end
    globalRec  = TP / (TP + FN); if (TP + FN) == 0, globalRec = 1; end
    globalF1   = 2 * (globalPrec * globalRec) / (globalPrec + globalRec);
    if (globalPrec + globalRec) == 0, globalF1 = 0; end

    fprintf('\nGLOBAL PERFORMANCE\n');
    fprintf('Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f\n', ...
        globalAcc, globalPrec, globalRec, globalF1);

end

%% --- EXPORT RESULTS ---
if ~isempty(summaryResults)

    if ~exist(PlotPath,'dir')
        mkdir(PlotPath);
    end

    csvFileName = fullfile(PlotPath,'MStra_WearGait_Results.csv');
    writetable(summaryResults,csvFileName);

    fprintf('\nResults saved to:\n%s\n', csvFileName);

end
