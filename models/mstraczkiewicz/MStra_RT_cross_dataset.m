 %% --- MStra RT Cross-Dataset Evaluation ---
% Runs the real-time MStra method across multiple datasets and saves plots to:
% outputs/plots/<dataset>/SigPro/

clear; clc; close all;
warning('off', 'MATLAB:datetime:AmbiguousInputFormat');

scriptDir = get_script_dir();
projectRoot = find_project_root(scriptDir);
outputsRoot = fullfile(projectRoot, 'outputs');
resultsDir = fullfile(outputsRoot, 'results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

allResults = table();

fprintf('\n============================================================\n');
fprintf('MStra RT Cross-Dataset Evaluation\n');
fprintf('Project root: %s\n', projectRoot);
fprintf('============================================================\n');

% Dataset order mirrors cross-dataset workflows used in Python scripts.
allResults = [allResults; evaluate_qsense_group(projectRoot, outputsRoot)];
allResults = [allResults; evaluate_weargait_group(projectRoot, outputsRoot)];
allResults = [allResults; evaluate_wisdm(projectRoot, outputsRoot)];
allResults = [allResults; evaluate_hmp(projectRoot, outputsRoot)];
allResults = [allResults; evaluate_freeliving(projectRoot, outputsRoot)];
allResults = [allResults; evaluate_bioclite(projectRoot, outputsRoot)];

if isempty(allResults)
    fprintf('\nNo evaluable files found.\n');
    return;
end

resultsCsv = fullfile(resultsDir, 'sigpro_mstra_rt_cross_dataset_results.csv');
writetable(allResults, resultsCsv);

fprintf('\nSaved per-record results to: %s\n', resultsCsv);

summaryTbl = compute_group_summary(allResults, {'Dataset'});
subjectTbl = compute_group_summary(allResults, {'Dataset', 'Subject'});

summaryCsv = fullfile(resultsDir, 'sigpro_mstra_rt_cross_dataset_summary.csv');
subjectCsv = fullfile(resultsDir, 'sigpro_mstra_rt_cross_dataset_by_subject.csv');
writetable(summaryTbl, summaryCsv);
writetable(subjectTbl, subjectCsv);

fprintf('Saved dataset summary to: %s\n', summaryCsv);
fprintf('Saved subject summary to: %s\n', subjectCsv);

plot_dataset_roc_pr_curves(allResults, outputsRoot);
fprintf('Saved ROC/PR curves to per-dataset SigPro folders.\n');

fprintf('\n================ Dataset Summary ================\n');
disp(summaryTbl(:, {'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'EvaluatedSamples'}));


%% ============================ DATASET EVALUATORS ============================

function results = evaluate_qsense_group(projectRoot, outputsRoot)
    results = table();
    candidates = {
        fullfile(projectRoot, 'Datasets', 'QSense_data')
        fullfile(projectRoot, 'Datasets', 'QSense_data_edge')
        fullfile(projectRoot, 'Datasets', 'QSense_data_mixed')
        fullfile(projectRoot, 'Datasets', 'QSense_data_clinic')
    };

    for i = 1:numel(candidates)
        dataPath = candidates{i};
        if ~exist(dataPath, 'dir')
            continue;
        end

        [~, datasetName] = fileparts(dataPath);
        plotPath = ensure_sigpro_plot_dir(outputsRoot, datasetName);

        fprintf('\n--- QSense dataset: %s ---\n', datasetName);
        subDirs = dir(dataPath);
        subDirs = subDirs([subDirs.isdir] & ~ismember({subDirs.name}, {'.', '..'}));

        for d = 1:numel(subDirs)
            folderName = subDirs(d).name;
            folderPath = fullfile(dataPath, folderName);

            targets = {
                's1_1RW.txt', 'Right'
                's2_2LW.txt', 'Left'
            };

            for t = 1:size(targets, 1)
                fname = targets{t, 1};
                wrist = targets{t, 2};
                fpath = fullfile(folderPath, fname);
                if ~isfile(fpath)
                    continue;
                end

                try
                    rec = load_qsense_record(fpath, folderName);
                    fs = 50;
                    params = get_rt_params(datasetName, fs);
                    row = run_record_and_plot(rec, datasetName, folderName, wrist, fs, params, plotPath);
                    if ~isempty(row)
                        results = [results; row];
                    end
                catch ME
                    fprintf('  [QSense] %s_%s ERROR: %s\n', folderName, wrist, ME.message);
                end
            end
        end
    end
end


function results = evaluate_weargait_group(projectRoot, outputsRoot)
    results = table();
    datasets = {
        fullfile(projectRoot, 'Datasets', 'WearGait', 'WearGait-PD')
        fullfile(projectRoot, 'Datasets', 'WearGait', 'WearGait-Ctrl')
    };

    for i = 1:numel(datasets)
        dataPath = datasets{i};
        if ~exist(dataPath, 'dir')
            continue;
        end

        datasetName = 'WearGait';
        plotPath = ensure_sigpro_plot_dir(outputsRoot, datasetName);

        files = dir(fullfile(dataPath, '*.csv'));
        for k = 1:numel(files)
            fname = files(k).name;
            lname = lower(fname);
            if ~contains(lname, 'freewalk')
                continue;
            end
            if contains(lname, 'manifest') || contains(lname, 'demographic')
                continue;
            end

            fpath = fullfile(files(k).folder, fname);
            try
                wg = load_weargait_data(fpath);
                subject = erase(fname, '.csv');

                if isfield(wg, 'right') && ~isempty(wg.right.acc_x)
                    rec = make_weargait_record(wg.right, subject, 'Right');
                    fs = estimate_fs(rec.time, 100);
                    params = get_rt_params(datasetName, fs);
                    [actStarts, actLbls] = extract_activity_markers(rec.time, rec.activities);
                    plotMeta = struct('activityStarts', actStarts, 'activityLabels', {actLbls});
                    row = run_record_and_plot(rec, datasetName, subject, 'Right', fs, params, plotPath, plotMeta);
                    if ~isempty(row), results = [results; row]; end
                end

                if isfield(wg, 'left') && ~isempty(wg.left.acc_x)
                    rec = make_weargait_record(wg.left, subject, 'Left');
                    fs = estimate_fs(rec.time, 100);
                    params = get_rt_params(datasetName, fs);
                    [actStarts, actLbls] = extract_activity_markers(rec.time, rec.activities);
                    plotMeta = struct('activityStarts', actStarts, 'activityLabels', {actLbls});
                    row = run_record_and_plot(rec, datasetName, subject, 'Left', fs, params, plotPath, plotMeta);
                    if ~isempty(row), results = [results; row]; end
                end
            catch ME
                fprintf('  [WearGait] %s ERROR: %s\n', fname, ME.message);
            end
        end
    end
end


function results = evaluate_wisdm(projectRoot, outputsRoot)
    results = table();
    dataPath = fullfile(projectRoot, 'Datasets', 'wisdm-dataset', 'raw', 'watch', 'accel');
    if ~exist(dataPath, 'dir')
        return;
    end

    datasetName = 'WISDM';
    plotPath = ensure_sigpro_plot_dir(outputsRoot, datasetName);
    files = dir(fullfile(dataPath, '*.txt'));

    wisdmActivityMap = containers.Map( ...
        {'A','B','C','D','E','F','G','H','I','J','K','L','M','O','P','Q','R','S'}, ...
        {'Walk','Jog','Stairs','Sit','Stand','Type','Teeth','Soup','Chips','Pasta', ...
         'Drink','Sandwich','Kick','Catch','Dribble','Write','Clap','Fold'});

    for i = 1:numel(files)
        fpath = fullfile(files(i).folder, files(i).name);
        subject = erase(files(i).name, '.txt');

        try
            rec = load_wisdm_record(fpath);
            actNames = cellfun(@(a) get_map_value(wisdmActivityMap, strtrim(a), strtrim(a)), ...
                rec.activities, 'UniformOutput', false);
            [activityStarts, activityLabels] = extract_activity_markers(rec.time, actNames);
            plotMeta = struct('activityStarts', activityStarts, 'activityLabels', {activityLabels});
            fs = estimate_fs(rec.time, 20);
            params = get_rt_params(datasetName, fs);
            row = run_record_and_plot(rec, datasetName, subject, 'Watch', fs, params, plotPath, plotMeta);
            if ~isempty(row)
                results = [results; row];
            end
        catch ME
            fprintf('  [WISDM] %s ERROR: %s\n', files(i).name, ME.message);
        end
    end
end


function results = evaluate_hmp(projectRoot, outputsRoot)
    results = table();
    dataPath = fullfile(projectRoot, 'Datasets', 'HMP_Dataset');
    if ~exist(dataPath, 'dir')
        return;
    end

    datasetName = 'HMP';
    plotPath = ensure_sigpro_plot_dir(outputsRoot, datasetName);
    gaitActs = {'walk', 'climb_stairs', 'descend_stairs'};

    acts = dir(dataPath);
    acts = acts([acts.isdir] & ~ismember({acts.name}, {'.', '..'}));

    % Build subject-level collections across all activities.
    subjectActivities = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for a = 1:numel(acts)
        actName = acts(a).name;
        if contains(lower(actName), '_model')
            continue;
        end

        actPath = fullfile(dataPath, actName);
        txtFiles = dir(fullfile(actPath, '*.txt'));
        isGait = any(strcmpi(actName, gaitActs));

        subjectFiles = containers.Map('KeyType', 'char', 'ValueType', 'any');
        for i = 1:numel(txtFiles)
            [subjectId, ts] = extract_hmp_subject_id_and_timestamp(txtFiles(i).name);
            if isempty(subjectId)
                continue;
            end

            fileInfo = struct( ...
                'path', fullfile(txtFiles(i).folder, txtFiles(i).name), ...
                'name', txtFiles(i).name, ...
                'timestamp', ts ...
            );

            if ~isKey(subjectFiles, subjectId)
                subjectFiles(subjectId) = {fileInfo};
            else
                tmp = subjectFiles(subjectId);
                tmp{end+1} = fileInfo;
                subjectFiles(subjectId) = tmp;
            end
        end

        subjectIds = sort(subjectFiles.keys());
        for i = 1:numel(subjectIds)
            subjectId = subjectIds{i};
            fileInfos = subjectFiles(subjectId);
            tsVals = cellfun(@(s) posixtime(s.timestamp), fileInfos);
            [~, order] = sort(tsVals);
            fileInfos = fileInfos(order);

            try
                recs = cell(1, numel(fileInfos));
                for k = 1:numel(fileInfos)
                    recs{k} = load_hmp_record(fileInfos{k}.path, isGait, actName);
                end

                recAct = concatenate_records_with_gaps(recs, 1.0);
                if isempty(recAct.time)
                    continue;
                end

                actBlock = struct( ...
                    'activity', actName, ...
                    'timestamp', fileInfos{1}.timestamp, ...
                    'record', recAct ...
                );

                if ~isKey(subjectActivities, subjectId)
                    subjectActivities(subjectId) = {actBlock};
                else
                    tmpActs = subjectActivities(subjectId);
                    tmpActs{end+1} = actBlock;
                    subjectActivities(subjectId) = tmpActs;
                end
            catch ME
                fprintf('  [HMP] %s_%s ERROR: %s\n', subjectId, actName, ME.message);
            end
        end
    end

    subjectIds = sort(subjectActivities.keys());
    for i = 1:numel(subjectIds)
        subjectId = subjectIds{i};
        actBlocks = subjectActivities(subjectId);

        tsVals = cellfun(@(s) posixtime(s.timestamp), actBlocks);
        [~, order] = sort(tsVals);
        actBlocks = actBlocks(order);

        [rec, activityStarts, activityLabels] = concatenate_activity_blocks(actBlocks, 1.0);
        if isempty(rec.time)
            continue;
        end

        try
            fs = 32;
            params = get_rt_params(datasetName, fs);
            plotMeta = struct('activityStarts', activityStarts, 'activityLabels', {activityLabels});
            row = run_record_and_plot(rec, datasetName, subjectId, 'AllActivities', fs, params, plotPath, plotMeta);
            if ~isempty(row)
                results = [results; row];
            end
        catch ME
            fprintf('  [HMP] %s_AllActivities ERROR: %s\n', subjectId, ME.message);
        end
    end
end


function results = evaluate_freeliving(projectRoot, outputsRoot)
    results = table();
    dataPath = fullfile(projectRoot, 'Datasets', 'Free_living');
    if ~exist(dataPath, 'dir')
        return;
    end

    datasetName = 'Free_living';
    plotPath = ensure_sigpro_plot_dir(outputsRoot, datasetName);

    files = dir(fullfile(dataPath, '*_annotated.csv'));
    if isempty(files)
        files = dir(fullfile(dataPath, '**', '*_annotated.csv'));
    end

    for i = 1:numel(files)
        fpath = fullfile(files(i).folder, files(i).name);
        recId = erase(files(i).name, '.csv');
        subject = parse_freeliving_subject(recId);

        try
            rec = load_freeliving_record(fpath);
            fs = estimate_fs(rec.time, 50);
            params = get_rt_params(datasetName, fs);
            row = run_record_and_plot(rec, datasetName, subject, 'Wrist', fs, params, plotPath);
            if ~isempty(row)
                results = [results; row];
            end
        catch ME
            fprintf('  [Free_living] %s ERROR: %s\n', files(i).name, ME.message);
        end
    end
end


function results = evaluate_bioclite(projectRoot, outputsRoot)
    results = table();
    matPath = fullfile(projectRoot, 'Datasets', 'Bioclite', 'data_6activities_plain.mat');
    if ~isfile(matPath)
        return;
    end

    datasetName = 'Bioclite';
    plotPath = ensure_sigpro_plot_dir(outputsRoot, datasetName);

    try
        S = load(matPath);
        if ~isfield(S, 'Data_plain')
            fprintf('  [Bioclite] Data_plain not found in %s\n', matPath);
            return;
        end
        Data = S.Data_plain;
    catch ME
        fprintf('  [Bioclite] load error: %s\n', ME.message);
        return;
    end

    for i = 1:numel(Data)
        try
            rec = load_bioclite_trial(Data{i});
            fs = estimate_fs(rec.time, 50);
            params = get_rt_params(datasetName, fs);
            [actStarts, actLbls] = extract_activity_markers(rec.time, rec.activities);
            plotMeta = struct('activityStarts', actStarts, 'activityLabels', {actLbls});
            row = run_record_and_plot(rec, datasetName, rec.subject, 'Preferred', fs, params, plotPath, plotMeta);
            if ~isempty(row)
                results = [results; row];
            end
        catch ME
            fprintf('  [Bioclite] trial %d ERROR: %s\n', i, ME.message);
        end
    end
end


%% ============================= CORE RT EXECUTION =============================

function row = run_record_and_plot(rec, datasetName, subject, wrist, fs, params, plotPath, varargin)
    row = table();

    plotMeta = struct();
    if ~isempty(varargin)
        plotMeta = varargin{1};
    end

    if numel(rec.time) < max(10, round(2 * fs))
        return;
    end

    vm = sqrt(rec.acc(:,1).^2 + rec.acc(:,2).^2 + rec.acc(:,3).^2);
    [yPred, yScore, sampleValid, rt] = run_rt_sequence(vm, rec.time, fs, params);

    yTrue = rec.y_true(:);
    yPred = yPred(:);
    yScore = yScore(:);
    sampleValid = sampleValid(:);

    nMin = min([numel(yTrue), numel(yPred), numel(yScore), numel(sampleValid), numel(rec.time)]);
    if nMin < max(10, round(2 * fs))
        return;
    end

    yTrue = yTrue(1:nMin);
    yPred = yPred(1:nMin);
    yScore = yScore(1:nMin);
    sampleValid = sampleValid(1:nMin);
    t = rec.time(1:nMin);

    [acc, prec, recMetric, f1, tp, tn, fp, fn, evalCount] = compute_metrics(yTrue, yPred, sampleValid);
    if evalCount == 0
        return;
    end

    plotFile = save_rt_debug_plot(plotPath, datasetName, subject, wrist, t, yTrue, yPred, sampleValid, rt, params, acc, prec, recMetric, f1, plotMeta);
    stepCount = sum(diff([0; yPred == 1]) == 1);
    evalIdx = sampleValid & isfinite(yTrue) & isfinite(yScore);
    yTrueEval = double(yTrue(evalIdx));
    yScoreEval = double(yScore(evalIdx));

    row = table({datasetName}, {subject}, {wrist}, acc, prec, recMetric, f1, tp, tn, fp, fn, ...
        nMin, evalCount, stepCount, {plotFile}, {yTrueEval}, {yScoreEval}, ...
        'VariableNames', {'Dataset','Subject','Wrist','Accuracy','Precision','Recall','F1', ...
                          'TP','TN','FP','FN','NumSamples','EvaluatedSamples','Steps','PlotFile','YTrueEval','YScoreEval'});

    fprintf('  %-12s %-12s | Acc=%.3f Prec=%.3f Rec=%.3f F1=%.3f\n', datasetName, [subject '_' wrist], acc, prec, recMetric, f1);
end


function [yPred, yScore, sampleValid, rt] = run_rt_sequence(vm, timeVec, fs, params)
    n = numel(vm);
    yPred = zeros(n, 1);
    yScore = zeros(n, 1);

    windowSize = max(4, round(2 * fs));
    stepSize = max(1, round(1 * fs));
    maxGap = 1.5 / fs;

    circularBuffer = ones(windowSize, 1) * vm(1);
    detectionState = 0;
    samplesSinceReset = 0;

    rt.t = [];
    rt.peakF = [];
    rt.maxPk = [];
    rt.ampVal = [];
    rt.validWindow = [];

    for s = 2:n
        dt = timeVec(s) - timeVec(s - 1);
        if ~isfinite(dt) || dt > maxGap || dt <= 0
            circularBuffer(:) = vm(s);
            detectionState = 0;
            samplesSinceReset = 0;
            continue;
        end

        samplesSinceReset = samplesSinceReset + 1;
        circularBuffer = [circularBuffer(2:end); vm(s)];

        if mod(s, stepSize) == 0 && s >= windowSize
            [isGait, newState, m] = run_mstra_rt(circularBuffer, fs, params.F_MIN, params.F_MAX, ...
                params.P_MIN, params.P_MAX, params.A_MIN, params.A_MAX, detectionState);
            detectionState = newState;
            confScore = compute_confidence_score(m, params);

            i1 = max(1, s - stepSize + 1);
            yPred(i1:s) = double(isGait);
            yScore(i1:s) = confScore;

            rt.t(end+1,1) = timeVec(s);
            rt.peakF(end+1,1) = m.peakF;
            rt.maxPk(end+1,1) = m.maxPk;
            rt.ampVal(end+1,1) = m.ampVal;
            rt.validWindow(end+1,1) = samplesSinceReset >= windowSize;
        end
    end

    sampleValid = false(n, 1);
    sCount = 0;
    for k = 2:n
        dt = timeVec(k) - timeVec(k - 1);
        if ~isfinite(dt) || dt > maxGap || dt <= 0
            sCount = 0;
        else
            sCount = sCount + 1;
        end
        sampleValid(k) = sCount >= windowSize;
    end
end


function [acc, prec, rec, f1, tp, tn, fp, fn, evalCount] = compute_metrics(yTrue, yPred, sampleValid)
    evalIdx = sampleValid;
    if ~any(evalIdx)
        acc = 0; prec = 0; rec = 0; f1 = 0;
        tp = 0; tn = 0; fp = 0; fn = 0;
        evalCount = 0;
        return;
    end

    yt = yTrue(evalIdx);
    yp = yPred(evalIdx);
    tp = sum(yt == 1 & yp == 1);
    tn = sum(yt == 0 & yp == 0);
    fp = sum(yt == 0 & yp == 1);
    fn = sum(yt == 1 & yp == 0);
    evalCount = numel(yt);

    acc = (tp + tn) / max(1, tp + tn + fp + fn);
    prec = tp / max(1, tp + fp);
    rec = tp / max(1, tp + fn);
    if (prec + rec) > 0
        f1 = 2 * (prec * rec) / (prec + rec);
    else
        f1 = 0;
    end
end


function outFile = save_rt_debug_plot(plotPath, datasetName, subject, wrist, t, yTrue, yPred, sampleValid, rt, params, acc, prec, rec, f1, plotMeta)
    outFile = '';
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [60, 60, 1150, 880]);
    ax = gobjects(4, 1);

    validWin = logical(rt.validWindow);
    c = [0.0 0.447 0.741];

    ax(1) = subplot(4,1,1); hold on;
    tp1 = rt.t; yf = rt.peakF;
    tp1(~validWin) = NaN; yf(~validWin) = NaN;
    plot(tp1, yf, 'Color', c, 'LineWidth', 1.2);
    yline(params.F_MIN, 'r--');
    if isfinite(params.F_MAX), yline(params.F_MAX, 'r--'); end
    ylabel('Peak Freq (Hz)');
    title('Criterion 1: Dominant Frequency');
    grid on;

    ax(2) = subplot(4,1,2); hold on;
    tp2 = rt.t; yp2 = rt.maxPk;
    tp2(~validWin) = NaN; yp2(~validWin) = NaN;
    plot(tp2, yp2, 'Color', c, 'LineWidth', 1.2);
    yline(params.P_MIN, 'r--');
    if isfinite(params.P_MAX), yline(params.P_MAX, 'r--'); end
    ylabel('Power');
    title('Criterion 2: Spectral Power');
    grid on;

    ax(3) = subplot(4,1,3); hold on;
    tp3 = rt.t; yp3 = rt.ampVal;
    tp3(~validWin) = NaN; yp3(~validWin) = NaN;
    plot(tp3, yp3, 'Color', c, 'LineWidth', 1.2);
    yline(params.A_MIN, 'r--');
    if isfinite(params.A_MAX), yline(params.A_MAX, 'r--'); end
    ylabel('Std(vm)');
    title('Criterion 3: Time-Domain Amplitude');
    grid on;

    ax(4) = subplot(4,1,4); hold on;
    yT = double(yTrue);
    yP = double(yPred);
    yT(~sampleValid) = NaN;
    yP(~sampleValid) = NaN;
    area(t, yT, 'FaceColor', c, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    stairs(t, yP, 'Color', c, 'LineWidth', 1.4);
    ylim([-0.1 1.1]);
    ylabel('Gait 0/1');
    xlabel('Time (s)');
    title(sprintf('RT Detection | Acc=%.2f Prec=%.2f Rec=%.2f F1=%.2f', acc, prec, rec, f1));
    legend({'GT', 'Pred'}, 'Location', 'northeast');
    grid on;

    % Optional activity boundary markers for appended records.
    if ~isempty(plotMeta) && isfield(plotMeta, 'activityStarts')
        starts = plotMeta.activityStarts(:);
        labels = {};
        if isfield(plotMeta, 'activityLabels')
            labels = plotMeta.activityLabels;
        end

        markColor = [0.35, 0.35, 0.35];
        for i = 1:numel(starts)
            x0 = starts(i);

            h1 = xline(ax(1), x0, '--', 'Color', markColor, 'LineWidth', 0.8);
            h1.HandleVisibility = 'off';
            h2 = xline(ax(2), x0, '--', 'Color', markColor, 'LineWidth', 0.8);
            h2.HandleVisibility = 'off';
            h3 = xline(ax(3), x0, '--', 'Color', markColor, 'LineWidth', 0.8);
            h3.HandleVisibility = 'off';

            if i <= numel(labels) && ~isempty(labels{i})
                h4 = xline(ax(4), x0, '--', char(labels{i}), 'Color', markColor, 'LineWidth', 0.8);
            else
                h4 = xline(ax(4), x0, '--', 'Color', markColor, 'LineWidth', 0.8);
            end
            h4.HandleVisibility = 'off';
        end
    end

    linkaxes(ax, 'x');
    sgtitle(sprintf('%s | %s | %s', datasetName, subject, wrist), 'Interpreter', 'none');

    safeName = regexprep(sprintf('%s_%s_%s_RT.png', datasetName, subject, wrist), '[^a-zA-Z0-9_\-\.]', '_');
    outFile = fullfile(plotPath, safeName);
    save_figure_png(fig, outFile, 200);
    close(fig);
end


function params = get_rt_params(datasetName, fs)
    params = struct();
    params.F_MIN = 0.5;
    params.F_MAX = 3.5;
    params.P_MIN = 3.0;
    params.P_MAX = inf;
    params.A_MIN = 0.05;
    params.A_MAX = inf;

    d = lower(datasetName);
    if contains(d, 'freeliving') || contains(d, 'free_living')
        params.F_MIN = 0.96248;
        params.F_MAX = 5.4359;
        params.P_MIN = 0.78589;
        params.P_MAX = 260.11;
        params.A_MIN = 0.035257;
        params.A_MAX = 0.12653;
    end

    % Keep frequency upper bound below Nyquist when low sample-rate files are found.
    nyq = fs / 2;
    params.F_MAX = min(params.F_MAX, max(params.F_MIN + 0.1, nyq - 0.05));
end


function score = compute_confidence_score(metrics, params)
    sFreq = bounded_score(metrics.peakF, params.F_MIN, params.F_MAX);
    sPower = bounded_score(metrics.maxPk, params.P_MIN, params.P_MAX);
    sAmp = bounded_score(metrics.ampVal, params.A_MIN, params.A_MAX);
    score = max(0, min(1, (sFreq + sPower + sAmp) / 3));
end


function s = bounded_score(x, lo, hi)
    x = double(x);
    lo = double(lo);
    hi = double(hi);

    if ~isfinite(x)
        s = 0;
        return;
    end

    if isfinite(hi)
        if x >= lo && x <= hi
            s = 1;
        elseif x < lo
            span = max(lo, 1e-6);
            s = max(0, 1 - (lo - x) / span);
        else
            span = max(hi, 1e-6);
            s = max(0, 1 - (x - hi) / span);
        end
    else
        if x >= lo
            scale = max(lo, 1e-6);
            s = min(1, 0.5 + 0.5 * (1 - exp(-(x - lo) / scale)));
        else
            s = max(0, x / max(lo, 1e-6));
        end
    end
end


%% ================================ LOADERS ====================================

function rec = load_qsense_record(filePath, folderName)
    opts = detectImportOptions(filePath);
    opts.VariableNamingRule = 'preserve';
    if numel(opts.VariableTypes) >= 2
        opts.VariableTypes{1} = 'char';
        opts.VariableTypes{2} = 'char';
    end
    T = readtable(filePath, opts);

    dateCol = string(T{:, 1});
    timeCol = string(T{:, 2});
    dt = datetime(dateCol + " " + timeCol, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSS');
    hasTime = ~all(isnat(dt));

    if hasTime
        % Remove backwards-jump blocks (buffer re-dumps).
        [dt, keep] = monotonic_keep(dt);
        T = T(keep, :);

        % Fix unrealistic clock jumps while preserving nominal sample interval.
        timeDiffs = diff(dt);
        jumpIdx = find(abs(timeDiffs) > days(100));
        nominalStep = seconds(1 / 50);
        for j = 1:numel(jumpIdx)
            idx = jumpIdx(j);
            falseGap = timeDiffs(idx) - nominalStep;
            dt(idx+1:end) = dt(idx+1:end) - falseGap;
            timeDiffs = diff(dt);
        end

        % Sort and remove repeats after correction.
        [dt, sortIdx] = sort(dt);
        T = T(sortIdx, :);
        [dt, idxUnique] = unique(dt);
        T = T(idxUnique, :);
        t = seconds(dt - dt(1));
    else
        t = (0:height(T)-1)' / 50;
    end

    [ax, ay, az] = get_acc_columns(T, [6 7 8]);
    valid = isfinite(ax) & isfinite(ay) & isfinite(az) & isfinite(t);
    t = t(valid);
    acc = [ax(valid), ay(valid), az(valid)];

    y = zeros(numel(t), 1);
    labelIdx = find(strcmpi(T.Properties.VariableNames, 'Label'), 1);
    if isempty(labelIdx)
        labelIdx = find(strcmpi(T.Properties.VariableNames, 'label'), 1);
    end
    if ~isempty(labelIdx)
        rawY = str2double(string(T{:, labelIdx}));
        rawY(isnan(rawY)) = 0;
        y = double(rawY(valid) > 0);
    else
        gaitFolder = contains(lower(folderName), {'walking', 'stairs'});
        y(:) = double(gaitFolder);
    end

    rec = struct('time', t(:), 'acc', acc, 'y_true', y(:));
end


function rec = make_weargait_record(sideStruct, subject, wrist)
    x = sideStruct.acc_x(:);
    y = sideStruct.acc_y(:);
    z = sideStruct.acc_z(:);
    t = sideStruct.time(:);
    labels = lower(string(sideStruct.labels(:)));

    yTrue = contains(labels, {'walk', 'jog', 'run', 'stair', 'climb', 'freewalk', 'gait'});
    valid = isfinite(x) & isfinite(y) & isfinite(z) & isfinite(t);

    rec = struct();
    rec.time = t(valid);
    rec.acc = [x(valid), y(valid), z(valid)];
    rec.y_true = double(yTrue(valid));
    rec.subject = subject;
    rec.wrist = wrist;
    rec.activities = cellstr(labels(valid));
end


function rec = load_wisdm_record(filePath)
    opts = delimitedTextImportOptions('NumVariables', 6);
    opts.Delimiter = ',';
    opts.VariableNames = {'Subject', 'Activity', 'Time', 'Acc_X', 'Acc_Y', 'Acc_Z'};
    opts.VariableTypes = {'double', 'string', 'double', 'double', 'double', 'string'};
    T = readtable(filePath, opts);

    z = str2double(strrep(string(T.Acc_Z), ';', ''));
    x = T.Acc_X;
    y = T.Acc_Y;
    tt = (T.Time - T.Time(1)) / 1e9;

    valid = isfinite(x) & isfinite(y) & isfinite(z) & isfinite(tt);
    acts = string(T.Activity(valid));
    yTrue = ismember(acts, {'A', 'C'});

    % Use synthetic monotonic time so all activity segments are appended
    % without timestamp gaps between activities.
    N = sum(valid);
    tt_synth = (0:N-1)' / 20.0;  % WISDM watch = 20 Hz

    rec = struct('time', tt_synth, 'acc', [x(valid), y(valid), z(valid)], ...
        'y_true', double(yTrue), 'activities', {cellstr(acts)});
end


function rec = load_hmp_record(filePath, isGait, activity)
    X = readmatrix(filePath);
    if isempty(X) || size(X, 2) < 3
        error('Invalid HMP file format');
    end
    X = X(:, 1:3);
    valid = all(isfinite(X), 2);
    X = X(valid, :);

    % HMP manual conversion: map [0..63] to [-14.709..+14.709], then median filter.
    X = -14.709 + (X ./ 63) * (2 * 14.709);
    n = 3;
    X(:, 1) = medfilt1(X(:, 1), n);
    X(:, 2) = medfilt1(X(:, 2), n);
    X(:, 3) = medfilt1(X(:, 3), n);

    fs = 32;
    t = (0:size(X, 1)-1)' / fs;
    yTrue = double(isGait) * ones(size(X, 1), 1);

    rec = struct('time', t, 'acc', X, 'y_true', yTrue, 'activity', activity);
end


function rec = concatenate_records_with_gaps(records, gapSec)
    if isempty(records)
        rec = struct('time', [], 'acc', [], 'y_true', []);
        return;
    end

    timeAll = [];
    accAll = [];
    yAll = [];
    currentOffset = 0;

    for i = 1:numel(records)
        r = records{i};
        if isempty(r.time)
            continue;
        end

        t = r.time(:);
        if isempty(timeAll)
            tAdj = t;
        else
            tAdj = currentOffset + gapSec + t;
        end

        timeAll = [timeAll; tAdj];
        accAll = [accAll; r.acc];
        yAll = [yAll; r.y_true(:)];
        currentOffset = tAdj(end);
    end

    rec = struct('time', timeAll, 'acc', accAll, 'y_true', yAll);
end


function [rec, activityStarts, activityLabels] = concatenate_activity_blocks(actBlocks, gapSec)
    activityStarts = [];
    activityLabels = {};

    if isempty(actBlocks)
        rec = struct('time', [], 'acc', [], 'y_true', []);
        return;
    end

    timeAll = [];
    accAll = [];
    yAll = [];
    currentOffset = 0;

    for i = 1:numel(actBlocks)
        b = actBlocks{i};
        r = b.record;
        if isempty(r.time)
            continue;
        end

        t = r.time(:);
        if isempty(timeAll)
            tAdj = t;
        else
            tAdj = currentOffset + gapSec + t;
        end

        activityStarts(end+1, 1) = tAdj(1); %#ok<AGROW>
        activityLabels{end+1, 1} = b.activity; %#ok<AGROW>

        timeAll = [timeAll; tAdj]; %#ok<AGROW>
        accAll = [accAll; r.acc]; %#ok<AGROW>
        yAll = [yAll; r.y_true(:)]; %#ok<AGROW>
        currentOffset = tAdj(end);
    end

    rec = struct('time', timeAll, 'acc', accAll, 'y_true', yAll);
end


function rec = load_freeliving_record(filePath)
    opts = detectImportOptions(filePath);
    opts.VariableNamingRule = 'preserve';
    T = readtable(filePath, opts);

    ax = read_col(T, {'ax', 'Acc_X', 'acc_x'}, []);
    ay = read_col(T, {'ay', 'Acc_Y', 'acc_y'}, []);
    az = read_col(T, {'az', 'Acc_Z', 'acc_z'}, []);
    if isempty(ax) || isempty(ay) || isempty(az)
        error('Missing accelerometer columns (ax/ay/az)');
    end

    timeVals = read_col(T, {'time', 'Time', 'timestamp'}, []);
    t = [];
    if ~isempty(timeVals)
        try
            dt = datetime(string(timeVals), 'InputFormat', 'MM/dd/yyyy HH:mm:ss.SSS', 'Locale', 'en_US');
            if ~all(isnat(dt))
                [dt, keep] = monotonic_keep(dt);
                ax = ax(keep); ay = ay(keep); az = az(keep);
                T = T(keep, :);
                [dt, idxUnique] = unique(dt);
                ax = ax(idxUnique); ay = ay(idxUnique); az = az(idxUnique);
                T = T(idxUnique, :);
                t = seconds(dt - dt(1));
            end
        catch
            t = [];
        end
    end
    if isempty(t)
        t = (0:numel(ax)-1)' / 50;
    end

    yRaw = read_col(T, {'Label', 'label', 'activity_label'}, []);
    if isempty(yRaw)
        error('Free_living record has no label column');
    end
    yTrue = str2double(string(yRaw));
    yTrue(isnan(yTrue)) = 0;
    yTrue = double(yTrue > 0);

    valid = isfinite(ax) & isfinite(ay) & isfinite(az) & isfinite(t);
    rec = struct('time', t(valid), 'acc', [ax(valid), ay(valid), az(valid)], 'y_true', yTrue(valid));
end


function rec = load_bioclite_trial(trial)
    if size(trial, 2) < 9
        error('Unexpected Bioclite trial shape');
    end

    tsMs = double(trial(:, 1));
    acc = double(trial(:, 2:4));
    participant = double(trial(1, 8));
    labels = double(trial(:, 9));

    t = (tsMs - tsMs(1)) / 1000.0;
    yTrue = double(labels == 6);
    valid = isfinite(t) & all(isfinite(acc), 2) & isfinite(yTrue);

    biocliteNames = {'Transitions','Spiral','Typing','Sitting','Beating','Brushing','Walking'};
    actNames = arrayfun(@(l) biocliteNames{min(max(round(l)+1, 1), 7)}, labels, 'UniformOutput', false);

    rec = struct();
    rec.time = t(valid);
    rec.acc = acc(valid, :);
    rec.y_true = yTrue(valid);
    rec.subject = sprintf('P%02d', max(1, round(participant)));
    rec.activities = actNames(valid);
end


%% ============================== SMALL HELPERS ================================

function plotPath = ensure_sigpro_plot_dir(outputsRoot, datasetName)
    plotPath = fullfile(outputsRoot, 'plots', datasetName, 'SigPro');
    if ~exist(plotPath, 'dir')
        mkdir(plotPath);
    end
end


function scriptDir = get_script_dir()
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
end


function projectRoot = find_project_root(startDir)
    projectRoot = startDir;
    for k = 1:10
        hasModels = exist(fullfile(projectRoot, 'models'), 'dir') == 7;
        hasDatasets = exist(fullfile(projectRoot, 'Datasets'), 'dir') == 7;
        if hasModels && hasDatasets
            return;
        end
        parentDir = fileparts(projectRoot);
        if strcmp(parentDir, projectRoot)
            break;
        end
        projectRoot = parentDir;
    end
    projectRoot = startDir;
end


function fs = estimate_fs(t, defaultFs)
    fs = defaultFs;
    if numel(t) < 5
        return;
    end
    dt = diff(t(:));
    dt = dt(isfinite(dt) & dt > 0);
    if isempty(dt)
        return;
    end
    medDt = median(dt);
    if medDt > 0
        fs = 1 / medDt;
    end
    fs = min(max(fs, 5), 250);
end


function [dtOut, keepMask] = monotonic_keep(dtIn)
    keepMask = true(numel(dtIn), 1);
    runningMax = dtIn(1);
    for i = 1:numel(dtIn)
        if dtIn(i) < runningMax
            keepMask(i) = false;
        else
            runningMax = dtIn(i);
        end
    end
    dtOut = dtIn(keepMask);
end


function [ax, ay, az] = get_acc_columns(T, fallbackIdx)
    ax = read_col(T, {'ax', 'accX', 'AccX', 'acc_x', 'Acc_X'}, fallbackIdx(1));
    ay = read_col(T, {'ay', 'accY', 'AccY', 'acc_y', 'Acc_Y'}, fallbackIdx(2));
    az = read_col(T, {'az', 'accZ', 'AccZ', 'acc_z', 'Acc_Z'}, fallbackIdx(3));

    ax = str2double(string(ax));
    ay = str2double(string(ay));
    az = str2double(string(az));
end


function v = read_col(T, names, fallbackIdx)
    v = [];
    vars = T.Properties.VariableNames;
    for i = 1:numel(names)
        idx = find(strcmpi(vars, names{i}), 1);
        if ~isempty(idx)
            v = T{:, idx};
            return;
        end
    end
    if ~isempty(fallbackIdx) && fallbackIdx <= width(T)
        v = T{:, fallbackIdx};
    end
end


function subject = parse_freeliving_subject(recId)
    subject = recId;
    parts = split(recId, '_');
    if numel(parts) >= 2
        subject = parts{2};
    end
end


function [activityStarts, activityLabels] = extract_activity_markers(timeVec, activities)
    % Returns start times and label names each time the activity label changes.
    activityStarts = [];
    activityLabels = {};
    if isempty(activities) || isempty(timeVec)
        return;
    end
    if ~iscell(activities)
        activities = cellstr(activities);
    end
    n = numel(activities);
    changes = [true; ~strcmp(activities(1:n-1), activities(2:n))];
    idx = find(changes);
    activityStarts = timeVec(idx);
    activityLabels = activities(idx);
end


function v = get_map_value(m, k, default)
    if isKey(m, k)
        v = m(k);
    else
        v = default;
    end
end


function [subjectId, ts] = extract_hmp_subject_id_and_timestamp(filename)
    subjectId = '';
    ts = datetime.empty;

    parts = split(erase(filename, '.txt'), '-');
    if numel(parts) < 9
        return;
    end

    subjectId = char(parts{end});
    try
        ts = datetime(sprintf('%s-%s-%s %s:%s:%s', ...
            parts{2}, parts{3}, parts{4}, parts{5}, parts{6}, parts{7}), ...
            'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    catch
        subjectId = '';
        ts = datetime.empty;
    end
end


function summaryTbl = compute_group_summary(T, groupVars)
    G = groupsummary(T, groupVars, 'sum', {'TP', 'TN', 'FP', 'FN', 'EvaluatedSamples'});

    tp = G.sum_TP;
    tn = G.sum_TN;
    fp = G.sum_FP;
    fn = G.sum_FN;

    precision = tp ./ max(1, tp + fp);
    recall = tp ./ max(1, tp + fn);
    accuracy = (tp + tn) ./ max(1, tp + tn + fp + fn);
    f1 = 2 .* (precision .* recall) ./ max(1e-9, precision + recall);

    summaryTbl = G(:, groupVars);
    summaryTbl.Accuracy = accuracy;
    summaryTbl.Precision = precision;
    summaryTbl.Recall = recall;
    summaryTbl.F1 = f1;
    summaryTbl.EvaluatedSamples = G.sum_EvaluatedSamples;
end


function plot_dataset_roc_pr_curves(T, outputsRoot)
    if isempty(T)
        return;
    end

    rocDir = fullfile(outputsRoot, 'plots', 'ROC', 'SigPro');
    prDir = fullfile(outputsRoot, 'plots', 'PR', 'SigPro');
    if ~ensure_dir_exists(rocDir, 'ROC')
        return;
    end
    if ~ensure_dir_exists(prDir, 'PR')
        return;
    end

    datasetNames = unique(string(T.Dataset), 'stable');

    figRoc = figure('Visible', 'off', 'Color', 'w', 'Position', [60, 60, 700, 580]);
    axRoc = axes(figRoc);
    hold(axRoc, 'on');
    hChance = plot(axRoc, [0 1], [0 1], 'k--', 'LineWidth', 1);

    figPr = figure('Visible', 'off', 'Color', 'w', 'Position', [60, 60, 700, 580]);
    axPr = axes(figPr);
    hold(axPr, 'on');

    rocHandles = hChance;
    rocLabels  = {'Chance'};
    prHandles  = gobjects(0);
    prLabels   = {};

    for i = 1:numel(datasetNames)
        ds = datasetNames(i);
        mask = string(T.Dataset) == ds;

        yTrue = [];
        yScore = [];
        yTrueCells  = T.YTrueEval(mask);
        yScoreCells = T.YScoreEval(mask);
        for k = 1:numel(yTrueCells)
            if isempty(yTrueCells{k}) || isempty(yScoreCells{k}), continue; end
            yTrue  = [yTrue;  double(yTrueCells{k}(:))];
            yScore = [yScore; double(yScoreCells{k}(:))];
        end

        valid  = isfinite(yTrue) & isfinite(yScore);
        yTrue  = yTrue(valid);
        yScore = yScore(valid);

        if numel(yTrue) < 2 || numel(unique(yTrue)) < 2
            fprintf('  Skipping ROC/PR for %s (need both classes present).\n', ds);
            continue;
        end

        [fpr, tpr, aucVal, youdenThresh]            = compute_roc_points(yTrue, yScore);
        [recall, precision, apVal, bestF1Thresh, bestF1Val] = compute_pr_points(yTrue, yScore);

        fprintf('  ROC %-20s AUC=%.3f, best Youden threshold=%.3f\n', ds, aucVal, youdenThresh);
        fprintf('  PR  %-20s AP=%.3f,  best F1=%.3f at threshold=%.3f\n', ds, apVal, bestF1Val, bestF1Thresh);

        h1 = plot(axRoc, fpr, tpr, 'LineWidth', 2);
        rocHandles(end+1) = h1; %#ok<AGROW>
        rocLabels{end+1}  = sprintf('%s (AUC=%.3f)', ds, aucVal); %#ok<AGROW>

        h2 = plot(axPr, recall, precision, 'LineWidth', 2);
        prHandles(end+1) = h2; %#ok<AGROW>
        prLabels{end+1}  = sprintf('%s (AP=%.3f)', ds, apVal); %#ok<AGROW>
    end

    % Finalise ROC subplot.
    xlabel(axRoc, 'False Positive Rate');
    ylabel(axRoc, 'True Positive Rate');
    title(axRoc, 'ROC Curves (MStra RT)');
    legend(axRoc, rocHandles, rocLabels, 'Location', 'southeast', 'Interpreter', 'none');
    grid(axRoc, 'on');
    axis(axRoc, [0 1 0 1]);

    % Finalise PR subplot.
    xlabel(axPr, 'Recall');
    ylabel(axPr, 'Precision');
    title(axPr, 'Precision-Recall Curves (MStra RT)');
    if ~isempty(prHandles)
        legend(axPr, prHandles, prLabels, 'Location', 'southwest', 'Interpreter', 'none');
    end
    grid(axPr, 'on');
    axis(axPr, [0 1 0 1]);

    rocFile = fullfile(rocDir, 'all_datasets_ROC.png');
    prFile = fullfile(prDir, 'all_datasets_PR.png');
    save_figure_png(figRoc, rocFile, 200);
    save_figure_png(figPr, prFile, 200);
    close(figRoc);
    close(figPr);
    fprintf('  Saved ROC figure: %s\n', rocFile);
    fprintf('  Saved PR figure: %s\n', prFile);
end


function ok = ensure_dir_exists(dirPath, tag)
    ok = true;
    if exist(dirPath, 'dir')
        return;
    end

    [mkOk, mkMsg] = mkdir(dirPath);
    ok = mkOk;
    if ~ok
        fprintf('  [%s] Cannot create output directory "%s": %s\n', tag, dirPath, mkMsg);
    end
end


function save_figure_png(fig, outFile, resolution)
    outDir = fileparts(outFile);
    if ~exist(outDir, 'dir')
        [mkOk, mkMsg] = mkdir(outDir);
        if ~mkOk
            error('Cannot create output directory "%s": %s', outDir, mkMsg);
        end
    end

    exportgraphics(fig, outFile, 'Resolution', resolution);
    if isfile(outFile)
        return;
    end

    try
        saveas(fig, outFile);
    catch
    end
    if isfile(outFile)
        return;
    end

    try
        print(fig, outFile, '-dpng', sprintf('-r%d', resolution));
    catch ME
        error('Figure export failed for "%s": %s', outFile, ME.message);
    end

    if ~isfile(outFile)
        error('Figure export failed for "%s" with no file created.', outFile);
    end
end

function [fpr, tpr, aucVal, youdenThresh] = compute_roc_points(yTrue, yScore)
    [scores, idx] = sort(yScore(:), 'descend');
    y = yTrue(idx) > 0;

    tp = cumsum(y == 1);
    fp = cumsum(y == 0);
    P = sum(y == 1);
    N = sum(y == 0);

    tprInner = tp / max(P, 1);
    fprInner = fp / max(N, 1);

    % Best Youden index = TPR - FPR.
    youden = tprInner - fprInner;
    [~, bestIdx] = max(youden);
    if ~isempty(bestIdx)
        youdenThresh = scores(bestIdx(1));
    else
        youdenThresh = NaN;
    end

    tpr = [0; tprInner; 1];
    fpr = [0; fprInner; 1];

    [fpr, uniqIdx] = unique(fpr, 'stable');
    tpr = tpr(uniqIdx);
    aucVal = trapz(fpr, tpr);

    if isempty(scores)
        aucVal = 0;
        youdenThresh = NaN;
    end
end


function [recall, precision, apVal, bestF1Thresh, bestF1Val] = compute_pr_points(yTrue, yScore)
    [scores, idx] = sort(yScore(:), 'descend');
    y = yTrue(idx) > 0;

    tp = cumsum(y == 1);
    fp = cumsum(y == 0);
    P = sum(y == 1);

    recallInner = tp / max(P, 1);
    precInner   = tp ./ max(tp + fp, 1);

    % Best F1 over all thresholds.
    denom = precInner + recallInner;
    f1Inner = zeros(size(denom));
    f1Inner(denom > 0) = 2 * precInner(denom > 0) .* recallInner(denom > 0) ./ denom(denom > 0);
    [bestF1Val, bestIdx] = max(f1Inner);
    if ~isempty(bestIdx)
        bestF1Thresh = scores(bestIdx(1));
    else
        bestF1Thresh = NaN;
        bestF1Val    = 0;
    end

    recall    = [0; recallInner; 1];
    precision = [1; precInner;   precInner(end)];

    % Make precision non-increasing (standard interpolation for AP).
    for i = numel(precision)-1:-1:1
        precision(i) = max(precision(i), precision(i+1));
    end

    apVal = trapz(recall, precision);
end


%% --- RT DETECTION FUNCTION (MStra_RT style) ---
function [finalDecision, newState, metrics] = run_mstra_rt(winData, fs, fMin, fMax, pMin, pMax, aMin, aMax, prevState)
    metrics.ampVal = std(winData);

    nfft = 512;
    w = hann(length(winData));
    winProc = (winData - mean(winData)) .* w;
    S = fft(winProc, nfft);
    P = abs(S(1:nfft/2+1)).^2;

    [metrics.maxPk, maxIdx] = max(P);
    freqs = fs * (0:(nfft/2)) / nfft;
    metrics.peakF = freqs(maxIdx);

    rawDecision = (metrics.peakF >= fMin && metrics.peakF <= fMax && ...
                   metrics.maxPk >= pMin && metrics.maxPk <= pMax && ...
                   metrics.ampVal >= aMin && metrics.ampVal <= aMax);

    newState = rawDecision;
    finalDecision = prevState & rawDecision;
end
