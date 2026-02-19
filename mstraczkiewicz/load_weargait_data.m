function data = load_weargait_data(filepath)

    opts = detectImportOptions(filepath);
    opts.VariableNamingRule = 'preserve';
    T = readtable(filepath, opts);

    % ----------------------------
    % Parse time column ("0.01 sec")
    % ----------------------------
    rawTime = T.Time;
    time = zeros(height(T),1);

    for i = 1:height(T)
        val = erase(rawTime{i},' sec');
        time(i) = str2double(val);
    end

    time = time - min(time);

    % ----------------------------
    % Extract both wrists
    % ----------------------------
    RX = str2double(string(T.("R_Wrist_Acc_X")));
    RY = str2double(string(T.("R_Wrist_Acc_Y")));
    RZ = str2double(string(T.("R_Wrist_Acc_Z")));

    LX = str2double(string(T.("L_Wrist_Acc_X")));
    LY = str2double(string(T.("L_Wrist_Acc_Y")));
    LZ = str2double(string(T.("L_Wrist_Acc_Z")));

    % ----------------------------
    % Extract ground truth
    % ----------------------------
    if ismember('GeneralEvent', T.Properties.VariableNames)
        labels = lower(string(T.GeneralEvent));
    else
        labels = repmat("unknown", height(T), 1);
    end

    % Remove NaNs separately per wrist
    validR = ~isnan(RX) & ~isnan(RY) & ~isnan(RZ);
    validL = ~isnan(LX) & ~isnan(LY) & ~isnan(LZ);

    data.time = time;

    data.right.acc_x = RX(validR);
    data.right.acc_y = RY(validR);
    data.right.acc_z = RZ(validR);
    data.right.time  = time(validR);
    data.right.labels = labels(validR);

    data.left.acc_x = LX(validL);
    data.left.acc_y = LY(validL);
    data.left.acc_z = LZ(validL);
    data.left.time  = time(validL);
    data.left.labels = labels(validL);

end
