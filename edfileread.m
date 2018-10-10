% needs data/xx.edf 
% needs edfread.m 

% makes /matdata/xx.mat files with headers "hdr" and data "record" of relevant channels
% saves it in unknown version, but could be specified

% "record" has the dimensions of [channel x samples]

% note the variable length of recordings, sample/128 equals recording time in sec

files = dir('data');
files = files(3:length(files)); 

for i = 1:length(files)

  [nothing,id,extension] = fileparts(files(i).name);

  [hdr, record] = edfread(strcat("data/",files(i).name),'targetSignals',[3:16]);
  mkdir matdata
  save(strcat('matdata/',id,'.mat'),'hdr','record');
  cd ..
  
endfor
