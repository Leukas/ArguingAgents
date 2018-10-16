% needs data/xx.edf 
% needs edfread.m 

% makes /matdata/xx.mat files with the data "record" of relevant channels, the digit the subject were thinking of "label", and subject id "subj"
% saves it in unknown version, but could be specified

% "record" has the dimensions of [channel x samples]

% note the variable length of recordings, sample/128 equals recording time in sec

files = dir('data');
files = files(3:length(files)); 

label = 0;
subj = 0;

for i = 1:length(files)

  [nothing,id,extension] = fileparts(files(i).name);

  [hdr, record] = edfread(strcat("data/",files(i).name),'targetSignals',[3:16]);
  
  label = id(length(id));
  subj = hdr.subjectID;
  
  mkdir matdata
<<<<<<< HEAD
  save(strcat('matdata/',id,'.mat'),'record', label, subj);
=======
  save(strcat('matdata/',id,'.mat'),'hdr','record');
>>>>>>> 6416b079f2d609cb85c5ed18903ab48ff7ca5e6b
  cd ..
  
endfor
