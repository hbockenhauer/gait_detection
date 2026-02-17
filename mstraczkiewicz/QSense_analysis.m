%% Analysis on self-recorded QSense Data
%% Mirroring WearGait-PD Reporting Style
clear; clc; close all;

% --- 1. CONFIGURATION ---
%dataPath = {'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_edge'};
dataPaths = {
   'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_edge'
   'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data'
};


% Use your optimized parameters from the grid search
F_MIN = 0.50;
F_MAX = 3.50;
P_THRESH = 3;
A_THRESH = 0.1; 

% --- 2. FILE DISCOVERY (Nested Subfolders) ---
% Find all subfolders (Subject_Activity)
summaryResults = table();
for d = 1:length(dataPaths)

    dataPath = dataPaths{d};
    fprintf('\nProcessing dataset: %s\n', dataPath);

    subDirs = dir(dataPath);
    subDirs = subDirs([subDirs.isdir] & ~ismember({subDirs.name}, {'.', '..'}));
    
    fprintf('F_MIN = %f\n', F_MIN);
    fprintf('F_MAX = %f\n', F_MAX);
    fprintf('P_THRESH = %f\n', P_THRESH);
    fprintf('A_THRESH = %f\n', A_THRESH);
    fprintf('Processing QSense folders (Checking s1_1RW and s2_2LW)...\n');
    fprintf('%-22s | %-8s | %-8s | %-8s | %-8s\n', 'Subject_Wrist', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
    fprintf('--------------------------------------------------------------------------------\n');
    
    % --- 3. MAIN PROCESSING LOOP ---
    for i = 1:length(subDirs)
        folderName = subDirs(i).name;
        folderPath = fullfile(dataPath, folderName);
        
        % Define the target wrist files
        targetFiles = {'s1_1RW.txt', 'Right'; 's2_2LW.txt', 'Left'};
        
        for t = 1:size(targetFiles, 1)
            fileName = targetFiles{t, 1};
            sideLabel = targetFiles{t, 2};
            fullFilePath = fullfile(folderPath, fileName);
            
            if ~isfile(fullFilePath), continue; end
            
            try
                % Load QSense Data
                opts = detectImportOptions(fullFilePath, 'FileType', 'text');
                opts.Delimiter = '\t';
                opts.VariableNamingRule = 'preserve';
                % Force time/date columns to text to avoid parsing errors
                opts.VariableTypes{1} = 'char'; opts.VariableTypes{2} = 'char';
                
                data = readtable(fullFilePath, opts);
                
                % Handle Time (Robust combination of Date and Time columns)
                try
                    timeStr = string(data{:,1}) + " " + string(data{:,2});
                    t_abs = datetime(timeStr); 
                    time = seconds(t_abs - t_abs(1));
                catch
                    % Fallback to 100Hz if datetime fails
                    time = (0:height(data)-1)' / 100;
                end
                
                % Extract Acceleration (Typically cols 6,7,8: accX, accY, accZ)
                % QSense standard indices: 6=accX, 7=accY, 8=accZ
                accX = data{:, 6}; accY = data{:, 7}; accZ = data{:, 8};
                
                % Clean Data
                validRows = ~isnan(accX) & ~isnan(accY) & ~isnan(accZ);
                vm = sqrt(accX(validRows).^2 + accY(validRows).^2 + accZ(validRows).^2);
                fs = round(1 / median(diff(time)));
                if isnan(fs) || fs < 1, fs = 100; end
    
                % Create Ground Truth based on Folder Name
                % Folder-level labeling: If 'walking' or 'stairs' in folder name, all is gait
                isGaitActivity = contains(lower(folderName), ["walk", "stairs"]);
                if isGaitActivity
                    y_true = ones(size(vm));
                else
                    y_true = zeros(size(vm));
                end
                
                % Run Detection (Using your lite function with optimized parameters)
                [y_pred, steps, peakF, ampVal, maxPk, T_vec] = run_straczkiewicz_optimized(vm, fs, F_MIN, F_MAX, P_THRESH, A_THRESH);

                % Plotting
                % --- 2. PLOTTING ---
                fig = figure('Name', sprintf('%s - %s', folderName, targetFiles{t,2}), 'Position', [100, 100, 1000, 800]);
                
                % Subplot 1: Frequency
                subplot(3,1,1);
                plot(T_vec, peakF, 'b', 'LineWidth', 1.5); hold on;
                yline(F_MIN, 'r--', 'F-Min'); yline(F_MAX, 'r--', 'F-Max');
                fill([T_vec(1) T_vec(end) T_vec(end) T_vec(1)], [F_MIN F_MIN F_MAX F_MAX], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
                ylabel('Peak Freq (Hz)'); title(['Debug Plot: ', folderName, ' (', targetFiles{t,2}, ')']);
                grid on; ylim([0 5]);

                % Subplot 2: Power & Amplitude
                subplot(3,1,2);
                yyaxis left
                plot(T_vec, maxPk, 'Color', [0 0.447 0.741]); ylabel('Spectrogram Power');
                yline(P_THRESH, 'LineStyle', ':', 'Color', [0 0.447 0.741], 'Label', 'P-Thresh');
                yyaxis right
                plot(T_vec, ampVal, 'Color', [0.85 0.325 0.098]); ylabel('StdDev Amplitude');
                yline(A_THRESH, 'LineStyle', ':', 'Color', [0.85 0.325 0.098], 'Label', 'A-Thresh');
                grid on;

                %Subplot 3: Final Decision vs Raw VM
                subplot(3,1,3);
                plot(time, vm - mean(vm), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5); hold on;
                stairs(time, y_pred * (max(vm)-mean(vm)), 'r', 'LineWidth', 2);
                ylabel('VM (Centered)'); xlabel('Time (s)');
                legend('Signal', 'Gait Detected'); grid on;
                
                % Calculate Metrics
                tp = sum(y_true == 1 & y_pred == 1);
                tn = sum(y_true == 0 & y_pred == 0);
                fp = sum(y_true == 0 & y_pred == 1);
                fn = sum(y_true == 1 & y_pred == 0);
                
                % Metric Safety checks
                acc = (tp + tn) / (tp + tn + fp + fn);
                prec = tp / (tp + fp); if (tp+fp)==0, prec = 1; end % Precision 1 if no detections
                rec = tp / (tp + fn); if (tp+fn)==0, rec = 1; end % Recall 1 if no gait to find
                f1 = 2 * (prec * rec) / (prec + rec); if (prec+rec)==0, f1 = 0; end
                
                % Store results
                fullID = sprintf('%s_%s', folderName, sideLabel);
                resRow = table({fullID}, {folderName}, {sideLabel}, acc, prec, rec, f1, steps, tp, tn, fp, fn, 'VariableNames', ...
                    {'ID','Subject','Wrist','Accuracy','Precision','Recall','F1','Steps','TP','TN','FP','FN'});

                summaryResults = [summaryResults; resRow];
                
                fprintf('%-22s | %-8.2f | %-8.2f | %-8.2f | %-8.2f\n', ...
                        fullID, acc, prec, rec, f1);
    
                % After processing a file
                stats = whos('data', 'S', 'vm');
                totalBytes = sum([stats.bytes]);
                %fprintf('Memory Load: %.2f MB\n', totalBytes / 1024^2);
                        
            catch ME
                fprintf('%-22s | ERROR: %s\n', folderName, ME.message);
            end
        end
    end
end

% --- CALCULATE SUMMARY STATISTICS ---
if ~isempty(summaryResults)

    TP = sum(summaryResults.TP);
    TN = sum(summaryResults.TN);
    FP = sum(summaryResults.FP);
    FN = sum(summaryResults.FN);

    globalAcc  = (TP + TN) / (TP + TN + FP + FN);
    globalPrec = TP / (TP + FP); if (TP+FP)==0, globalPrec = 1; end
    globalRec  = TP / (TP + FN); if (TP+FN)==0, globalRec = 1; end
    globalF1   = 2 * (globalPrec * globalRec) / (globalPrec + globalRec);
    if (globalPrec + globalRec)==0, globalF1 = 0; end

    fprintf('\n======================================================================\n');
    fprintf('GLOBAL PERFORMANCE (Combined Both Folders)\n');
    fprintf('----------------------------------------------------------------------\n');
    fprintf('Accuracy:  %.4f\n', globalAcc);
    fprintf('Precision: %.4f\n', globalPrec);
    fprintf('Recall:    %.4f\n', globalRec);
    fprintf('F1-Score:  %.4f\n', globalF1);

else
    disp('No data processed.');
end


%% --- OPTIMIZED DETECTION FUNCTION ---
function [wi, steps, peakFs, ampVals, maxPks, T_vec] = run_straczkiewicz_optimized(vm, fs, fMin, fMax, pThr, aThr)
    fs_int = round(fs);
    % Spectrogram
    [S, F, T_vec] = spectrogram(detrend(vm), 2*fs_int, fs_int, 512, fs);
    Cabs = abs(S).^2;

    numWindows = length(T_vec);
    peakFs  = zeros(1, numWindows);
    maxPks  = zeros(1, numWindows);
    ampVals = zeros(1, numWindows);
    wi_raw  = zeros(1, numWindows);
    
    for i = 1:numWindows
        % Time-domain Amplitude check (StdDev)
        t_center = T_vec(i);
        idx = round(t_center * fs);
        win_idx = max(1, idx-fs_int):min(length(vm), idx+fs_int);
        ampVals(i) = std(vm(win_idx));
        
        % Frequency Peak check
        [maxPks(i), maxIdx] = max(Cabs(:,i));
        peakFs(i) = F(maxIdx);
        
        if peakFs(i) >= fMin && peakFs(i) <= fMax && maxPks(i) > pThr && ampVals(i) > aThr
            wi_raw(i) = 1;
        end
    end
    
    % Refinement and Upsampling to signal length
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