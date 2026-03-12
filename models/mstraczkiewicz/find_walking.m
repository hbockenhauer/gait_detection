%function [wi, steps, cad] = find_walking(vm, fs, min_amp, T, delta, ...
%    alpha, beta, step_freq)
% Identify walking periods and its features from a raw accelerometry signal
% collected using wearable devices.
%
% Detailed method description was published in:
% Straczkiewicz M., Huang E., Onnela J.-P., A “one-size-fits-most” walking
% recognition method for smartphones, smartwatches, and wearable
% accelerometers, npj Digital Medicine, 2023.
%
% Inputs:
% vm ~      vector magnitude of raw acceleration signal
% fs ~      sampling frequency of data collection (in Hz, e.g., 10)
% min_amp ~ amplitude threshold (in g (gravitational units), e.g., 0.2)
% T  ~      minimum walking duration (in seconds, e.g., 3)
% delta ~   maximum difference between consecutive peaks (in multiplication
%           of 0.05Hz, e.g., 2)
% alpha ~   maximum ratio between dominant peak below and within step
%           frequency range (e.g., 0.6)
% beta ~    maximum ratio between dominant peak above and within step
%           frequency range (e.g., 2.5)
% step_freq ~ step frequency range (in Hz or steps per second, e.g., [1.4,
%           2.3])
%
% Outputs:
% wi ~      walking indication
% steps ~   total number of steps calculated from the input signal
% cad ~     temporal walking cadence (steps per second)
% 
% Example:
% find_walking(vm, 10, 0.2, 3, 2, 0.6, 2.5, [1.4, 2.3])
% 
%
% Script author:
% Marcin Straczkiewicz, PhD
% mstraczkiewicz@hsph.harvard.edu; mstraczkiewicz@gmail.com
%
% Last modification: 20/12/2022

function [wi, steps, cad] = find_walking(vm, fs, min_amp, T, delta, alpha, beta, step_freq)
% Identify walking periods and features from accelerometry signal
% Corrected to avoid non-integer size errors for QSense wrist data

% vectorize the input
vm = vm(:);

% truncate vm to full seconds
n_sec = floor(numel(vm)/fs);
vm = vm(1:n_sec*fs);

% preallocate memory
wi = zeros(size(vm));
cad = zeros(n_sec,1);
steps = 0;

% identify valid seconds based on min_amp
pp = peak2peak(reshape(vm, [fs, n_sec]))';
valid_sec = pp >= min_amp;
valid = repelem(valid_sec, fs);

% trim vm to valid samples
vm = vm(valid);
n_sec_valid = floor(numel(vm)/fs);  % recalc number of full valid seconds
if n_sec_valid == 0
    return
end

% smooth signal at its ends
w = tukeywin(numel(vm), 0.02);
vm = vm .* w;
vm = [zeros(5*fs,1); vm; zeros(5*fs,1)];
vm_len = numel(vm) - 10*fs; % for later indexing

% compute CWT
[Cima,freqs] = cwt(vm, fs, 'morse', ...
    'WaveletParameters', [3 90], ...
    'VoicesPerOctave', 48, ...
    'NumOctaves', 4);
Cima = Cima(:, 5*fs+1 : end-5*fs);  % remove padded edges
Cabs = abs(Cima).^2;

% interpolate over linear frequency domain
freqs_linspace = round(min(freqs),1):0.05:round(max(freqs),1);
Cabs = interp2(1:vm_len, freqs, Cabs, 1:vm_len, freqs_linspace', 'linear');

% truncate Cabs to full seconds
n_sec_cabs = floor(size(Cabs,2)/fs);
Cabs = Cabs(:,1:n_sec_cabs*fs);

% step frequency indices
[~, loc1] = min(abs(freqs_linspace - step_freq(1)));
[~, loc2] = min(abs(freqs_linspace - step_freq(2)));

% detect peaks
D = zeros(size(Cabs,1), n_sec_cabs);
for i = 1:n_sec_cabs
    vm_1s_start = (i-1)*fs + 1;
    vm_1s_finish = i*fs;

    signal = sum(Cabs(:, vm_1s_start:vm_1s_finish),2);
    [pks, pks_locs] = findpeaks(signal);           % get peak values and locations
    [~, I] = sort(pks,'descend');                  % sort peaks descending
    pks_locs = pks_locs(I);
    pks = pks(I);
    
    step_pks_locs = find(pks_locs >= loc1 & pks_locs <= loc2, 1, 'first');
    if isempty(step_pks_locs)
        continue
    end

    x = zeros(size(Cabs,1),1);
    if pks_locs(1) > loc2
        if pks(1)/pks(step_pks_locs(1)) < beta
            x(pks_locs(step_pks_locs(1))) = 1;
        end
    elseif pks_locs(1) < loc1
        if pks(1)/pks(step_pks_locs(1)) < alpha
            x(pks_locs(step_pks_locs(1))) = 1;
        end
    else
        x(pks_locs(step_pks_locs(1))) = 1;
    end
    D(:,i) = x;
end

% align peaks with valid seconds
E = zeros(size(D,1), n_sec);
E(:, valid_sec(1:n_sec)) = D(:,1:min(n_sec, size(D,2)));

% periodicity check
if T == 1
    e = sum(E,1);
else
    B = find_continuous_dominant_peaks(E, T, delta);
    e = sum(B,1);
end

% mark walking seconds
wi_sec = zeros(size(e));
wi_sec(e>0) = 1;

% stretch to sample level
wi = repelem(wi_sec, fs);

% temporal cadence
cad = zeros(size(e));
if T > 1
    for i = 1:numel(e)
        ind_freqs = find(B(:,i));
        if ~isempty(ind_freqs) && numel(ind_freqs) == 1
            cad(i) = freqs_linspace(ind_freqs);
        end
    end
end

% total steps
steps = sum(cad);
end
