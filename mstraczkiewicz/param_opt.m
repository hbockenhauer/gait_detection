%% --- 0. CONFIGURATION ---
fs = 50;
nTrials = 20000;          % Increase trials for better coverage
subset_walk_max = 20000; % Max walking samples
subset_non_max = 20000;  % Max non-walking samples

% Load your full vm_all and y_true_all first (from your previous code)

%% --- 1. BALANCED SUBSET FOR OPTIMIZATION ---
walk_idx = find(y_true_all==1);
nonwalk_idx = find(y_true_all==0);

n_walk = min(length(walk_idx), subset_walk_max);
n_non  = min(length(nonwalk_idx), subset_non_max);

subset_idx = [walk_idx(randperm(length(walk_idx), n_walk)); ...
              nonwalk_idx(randperm(length(nonwalk_idx), n_non))];

vm_subset = vm_all(subset_idx);
y_true_subset = y_true_all(subset_idx);

fprintf('Balanced subset: %d walking, %d non-walking\n', n_walk, n_non);

%% --- 2. DEFINE PARAMETER RANGES (physiologically realistic) ---
paramRanges.fMin        = [0.3, 1.];   % walking lower bound (Hz)
paramRanges.fMax        = [2.5, 4.5];       % walking upper bound (Hz)
paramRanges.pThr        = [0, 0.5];     % observed spectrogram power
paramRanges.aThr        = [0, 0.5];     % observed amplitude std dev
paramRanges.winLenSec   = [0.5, 2];     % window length in seconds
paramRanges.nfftVal     = [128, 1024];  % FFT length
paramRanges.consWindows = [1, 5];       % smoothing windows

bestF1 = -Inf;
bestParams = struct();

%% --- 3. RANDOM SEARCH LOOP ---
for trial = 1:nTrials
    % Sample random parameters
    params.fMin        = rand*(paramRanges.fMin(2)-paramRanges.fMin(1)) + paramRanges.fMin(1);
    params.fMax        = rand*(paramRanges.fMax(2)-paramRanges.fMax(1)) + paramRanges.fMax(1);
    if params.fMax <= params.fMin, params.fMax = params.fMin + 0.5; end % ensure fMax > fMin
    params.pThr        = rand*(paramRanges.pThr(2)-paramRanges.pThr(1)) + paramRanges.pThr(1);
    params.aThr        = rand*(paramRanges.aThr(2)-paramRanges.aThr(1)) + paramRanges.aThr(1);
    params.winLenSec   = rand*(paramRanges.winLenSec(2)-paramRanges.winLenSec(1)) + paramRanges.winLenSec(1);
    params.nfftVal     = randi(paramRanges.nfftVal);
    params.consWindows = randi(paramRanges.consWindows);

    % Run detection
    [y_pred, ~, ~, ~, ~, ~] = run_straczkiewicz_optimized(vm_subset, fs, ...
        params.fMin, params.fMax, params.pThr, params.aThr, ...
        params.winLenSec, params.nfftVal, params.consWindows);

    % Compute F1 score
    TP = sum((y_pred==1) & (y_true_subset==1));
    FP = sum((y_pred==1) & (y_true_subset==0));
    FN = sum((y_pred==0) & (y_true_subset==1));

    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1 = 2 * (precision * recall) / (precision + recall + eps);

    % Update best parameters
    if f1 > bestF1
        bestF1 = f1;
        bestParams = params;
        best_y_pred = y_pred; % save predictions of best trial
    end
end

%% --- 4. DISPLAY RESULTS ---
fprintf('Best F1 score on balanced subset: %.4f\n', bestF1);
disp('Best parameter set:');
disp(bestParams);

% %% --- 5. OPTIONAL: PLOT BEST TRIAL ---
% figure;
% stairs(vm_subset, best_y_pred, 'LineWidth', 1.5);
% xlabel('Sample'); ylabel('Detected Walking (1=Walk, 0=Non-walk)');
% title('Best Random Search Trial: Predicted Walking vs Time');
% grid on;


%% --- 5. UPDATE YOUR DETECTION FUNCTION ---
function [wi, steps, peakFs, ampVals, maxPks, T_vec] = run_straczkiewicz_optimized(vm, fs, fMin, fMax, pThr, aThr, winLenSec, nfftVal, consWindows)
    winSamples = round(winLenSec*fs);
    [S,F,T_vec] = spectrogram(detrend(vm), 2*winSamples, winSamples, nfftVal, fs);
    Cabs = abs(S).^2;

    numWindows = length(T_vec);
    peakFs = zeros(1,numWindows); maxPks = zeros(1,numWindows); ampVals = zeros(1,numWindows); wi_raw = zeros(1,numWindows);

    for i = 1:numWindows
        t_center = T_vec(i);
        idx = round(t_center * fs);
        win_idx = max(1, idx-winSamples):min(length(vm), idx+winSamples);
        ampVals(i) = std(vm(win_idx));
        [maxPks(i), maxIdx] = max(Cabs(:,i));
        peakFs(i) = F(maxIdx);

        if peakFs(i)>=fMin && peakFs(i)<=fMax && maxPks(i)>pThr && ampVals(i)>aThr
            wi_raw(i) = 1;
        end
    end

    wi_refined = movsum(wi_raw, [consWindows-1 0]) >= consWindows;
    wi = zeros(size(vm));
    for i = 1:length(T_vec)
        if wi_refined(i)
            idx = round(T_vec(i) * fs);
            wi(max(1, idx-winSamples):min(length(wi), idx)) = 1;
        end
    end

    steps = sum(wi_refined);
end