%% --- OPTIMIZED DETECTION FUNCTION ---
function [wi, steps, peakFs, ampVals, maxPks, T_vec] = run_straczkiewicz_optimized(vm, fs, fMin, fMax, pThr, aThr)
    fs_int = round(fs);
    [S, F, T_vec] = spectrogram(detrend(vm), 2*fs_int, fs_int, 512, fs);
    Cabs = abs(S).^2;
    numWindows = length(T_vec);
    peakFs = zeros(1, numWindows); maxPks = zeros(1, numWindows); ampVals = zeros(1, numWindows); wi_raw = zeros(1, numWindows);
    for i = 1:numWindows
        t_center = T_vec(i);
        idx = round(t_center * fs);
        win_idx = max(1, idx-fs_int):min(length(vm), idx+fs_int);
        ampVals(i) = std(vm(win_idx));
        [maxPks(i), maxIdx] = max(Cabs(:,i));
        peakFs(i) = F(maxIdx);
        if peakFs(i) >= fMin && peakFs(i) <= fMax && maxPks(i) > pThr && ampVals(i) > aThr
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