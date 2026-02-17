function [wi, steps, cad] = find_walking(vm, fs, min_amp, T, delta, ...
                                         alpha, beta, step_freq)

vm = vm(:);
num_secs = floor(numel(vm)/fs);
vm = vm(1:num_secs*fs);

%% 1. Amplitude mask (RELAXED)
nsec = floor(numel(vm)/fs);
if nsec < 1
    wi=zeros(size(vm)); steps=0; cad=zeros(num_secs,1); return
end

pp = peak2peak(reshape(vm(1:nsec*fs),fs,nsec)') ;
valid_sec = pp >= min_amp;

if ~any(valid_sec)
    wi=zeros(size(vm)); steps=0; cad=zeros(num_secs,1); return
end

vm_valid = vm(repelem(valid_sec(:),fs));


%% 2. Wavelet
[C,f] = cwt(vm_valid,fs,'morse','VoicesPerOctave',48);
P = abs(C).^2;

[f,idx]=sort(f); P=P(idx,:);
fgrid=(0:0.05:4)';                       % wrist-appropriate
P=interp1(f,P,fgrid,'linear',0);

[~,f1]=min(abs(fgrid-step_freq(1)));
[~,f2]=min(abs(fgrid-step_freq(2)));

%% 3. Per-second dominant peak
N=floor(size(P,2)/fs);
D=zeros(length(fgrid),N);

for i=1:N
    seg=sum(P(:,(i-1)*fs+1:i*fs),2);
    [pk,loc]=findpeaks(seg);
    if isempty(pk), continue; end
    [pk,ix]=sort(pk,'descend'); loc=loc(ix);

    k=find(loc>=f1 & loc<=f2,1);
    if isempty(k), continue; end

    if loc(1)>f2 && pk(1)/pk(k)<beta
        D(loc(k),i)=1;
    elseif loc(1)<f1 && pk(1)/pk(k)<alpha
        D(loc(k),i)=1;
    elseif loc(1)>=f1 && loc(1)<=f2
        D(loc(k),i)=1;
    end
end

%% 4. Temporal consistency
E=zeros(length(fgrid),num_secs);
E(:,valid_sec)=D;

if T>1
    B=find_continuous_dominant_peaks(E,T,delta);
else
    B=E;
end

%% 5. Outputs
wi=repelem(sum(B,1)>0,fs);
cad=zeros(num_secs,1);
for i=1:num_secs
    k=find(B(:,i),1);
    if ~isempty(k), cad(i)=fgrid(k); end
end
steps=round(sum(cad));

end
