% This is a script for extracting (.bzip2), reading (.ppm) and saving (.mat) data aquired from 
% https://www.nist.gov/itl/iad/image-group/color-feret-database 


%% Dependencies
%
% Octave (not tested for MATLAB compatibility)
% https://www.gnu.org/software/octave/
%
% pkg install -forge image
% pkg load image

%% 1 and 2 corresponds to dvd1 and dvd2
%% unzipped files dissapear when zipped

%% subject label example
% id=cfrS00001
% gender=Male
% yob=1943
% race=White

pkg load image

basedir = '/media/zalty/403F-6532/colorferet/colorferet/';
datadir = 'bunzip/';
slabeldir = '/data/ground_truths/name_value/';

basedir2 = '/home/zalty/Documents/DATA/colorferet/';
datadir = 'bunzip/';

subj1 = char(readdir([basedir, 'dvd1/data/thumbnails/']))(3:end,:);
subj2 = char(readdir([basedir, 'dvd2/data/thumbnails/']))(3:end,:);

slab_id = [];
slab_gender = [];
slab_year = [];
slab_race = [];

for k=1:size(subj1)(1)

  % unzipping files
  files1 = char(readdir([basedir, 'dvd1/data/thumbnails/',subj1(k,:)]))(3:end,:);
  for l=1:size(files1)(1)
  bunzip2([basedir, 'dvd1/data/thumbnails/',subj1(k,:),'/',files1(l,:)],basedir,datadir);
  endfor
    
  % get subject lables
  slab = textread([basedir, 'dvd1/data/ground_truths/name_value/',subj1(k,:),'/',subj1(k,:),'.txt'],'%s');
  slab_id = [slab_id; slab{1}(8:end)];
  slab_gender = [slab_gender; slab{2}(8)];
  slab_year = [slab_year; slab{3}(5:end)];
  slab_race =[slab_race; slab{4}(:)', blanks(50-length(slab{4}(:)'))];
  
endfor

% same but from dvd 2
for k=1:size(subj2)(1)
  
  files2 = char(readdir([basedir,'dvd2/data/thumbnails/',subj2(k,:)]))(3:end,:);
  for l=1:size(files2)(1)
  bunzip2([basedir,'dvd2/data/thumbnails/',subj2(k,:),'/',files2(l,:)],basedir,datadir);
  endfor
  
  slab = textread([basedir, 'dvd2/data/ground_truths/name_value/',subj2(k,:),'/',subj2(k,:),'.txt'],'%s');
  slab_id = [slab_id; slab{1}(8:end)];
  slab_gender = [slab_gender; slab{2}(8)];
  slab_year = [slab_year; slab{3}(5:end)];
  slab_race =[slab_race; slab{4}(:)', blanks(50-length(slab{4}(:)'))];
  
endfor


%%% read files and more labels

imfiles = readdir([basedir2,datadir])(3:end);
imnum={};
label={};
label_subj=[];
label_date=[];
label_pose=[];
%ppm_weird=[];

for k=1:size(imfiles)(1)
  
  %try
  %  imread([basedir, datadir, imfiles(k,:)]);
  %catch
  %  ppm_weird=[ppm_weird; imfiles(k,:)];
  %  continue
  %end_try_catch
  
  % read images, convert to grayscale
  imnum{length(imnum)+1}=rgb2gray(imread([basedir, datadir, imfiles{k}]));
  label{length(label)+1}=imfiles{k}(1:end-4);
  
  label_subj=[label_subj; label{k}(1:5)];
  label_date=[label_date; label{k}(7:12)];
  label_pose=[label_pose; label{k}(14:15)];
  
endfor

% get image labels
label_year = [];
label_race = [];
label_gender = [];

for k=1:length(label_subj)
  
  l = find(str2num(slab_id) == str2num(label_subj(k,:)));
  
  label_year = [label_year; slab_year(l,:)];
  label_race = [label_race; slab_race(l,:)];
  label_gender = [label_gender; slab_gender(l,:)];
  
endfor

save('-v7',[basedir,'imnum.mat'],'imnum');

save('-v7',[basedir,'label_year.mat'],'label_year');
save('-v7',[basedir,'label_subj.mat'],'label_subj');
save('-v7',[basedir,'label_race.mat'],'label_race');
save('-v7',[basedir,'label_pose.mat'],'label_pose');
save('-v7',[basedir,'label_gender.mat'],'label_gender');
save('-v7',[basedir,'label_date.mat'],'label_date');




