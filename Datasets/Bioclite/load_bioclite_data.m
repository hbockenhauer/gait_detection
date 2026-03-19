% Run this in MATLAB to convert the table format to a plain cell array
load('data_6activities.mat');   % loads data_6activities (20x1 cell of tables)

Data_plain = cell(20, 1);

for i = 1:20
    t = data_6activities{i};
    % Columns: ts_ms, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, participant_id, activity_label
    mat = [t.Var2, t.Var3, t.Var4, t.Var5, t.Var6, t.Var7, t.Var8, t.label1, t.label2];
    Data_plain{i} = mat;
end
save('data_6activities_plain.mat', 'Data_plain', '-v7');
disp('Done.');
