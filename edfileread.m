% This is a script for saving (.mat) relevant channels of EEG data (.edf) with labels.
%
% EEG data is from
% Tirupattur, P., Rawat, Y.S., Spampinato, C., & Shah, M. (2018). ThoughtViz: Visualizing Human Thoughts Using Generative Adversarial Network.
% Kumar, Pradeep & Saini, Rajkumar & Roy, Partha & Kumar Sahu, Pawan & Dogra, Debi. (2017). Envisioned speech recognition using EEG sensors. Personal and Ubiquitous Computing. 1-15. 10.1007/s00779-017-1083-4. 

%% Dependencies
%
% Octave (not tested for MATLAB compatibility)
% https://www.gnu.org/software/octave/
%
% edfread.m
% https://nl.mathworks.com/matlabcentral/fileexchange/31900-edfread

% New data can be found as "record", the digit the subject were thinking of as "label", and subject id "subj".
% "record" has the dimensions of [channel x samples]
% Note the variable length of recordings, sample/128 equals recording time in sec.

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
  save(strcat('matdata/',id,'.mat'),'record', label, subj);
  cd ..
  
endfor
