fid = fopen('1LE1_L.dat','r');
datacell = textscan(fid, '%u%f');
fclose(fid);
t_vec = datacell{1};
Z = datacell{2};
index = 1:length(Z);
