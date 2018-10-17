

% labels

% xxxxx.txt
% /media/zalty/403F-6532/colorferet/colorferet/dvd[1 2]/data/ground_truths/name_value/xxxxx

% example
% id=cfrS00001
% gender=Male
% yob=1943
% race=White

%% Dependencies
% pkg install -forge image
% pkg load image

%% 1 and 2 corresponds to dvd1 and dvd2
%% unzipped files dissapear when zipped

basedir = '/media/zalty/403F-6532/colorferet/colorferet/';
datadir = 'bunzip/';

subj1 = char(readdir([basedir, 'dvd1/data/thumbnails/']))(3:end,:);
subj2 = char(readdir([basedir, 'dvd2/data/thumbnails/']))(3:end,:);

for k=1:size(subj1)(1)
  files1 = char(readdir([basedir, 'dvd1/data/thumbnails/',subj1(k,:)]))(3:end,:);
  for l=1:size(files1)(1)
  bunzip2([basedir, 'dvd1/data/thumbnails/',subj1(k,:),'/',files1(l,:)],basedir,datadir);
  endfor
endfor


for k=1:size(subj2)(1)
  files2 = char(readdir([basedir,'dvd2/data/thumbnails/',subj2(k,:)]))(3:end,:);
  for l=1:size(files2)(1)
  bunzip2([basedir,'dvd2/data/thumbnails/',subj2(k,:),'/',files2(l,:)],basedir,datadir);
  endfor
endfor


%% read files
pkg load image

basedir = '/media/zalty/403F-6532/colorferet/colorferet/';
datadir = 'bunzip/';

imfiles = char(readdir([basedir,datadir]))(3:end,:);
imnum={};
ppm_weird=[];

for k=1:size(imfiles)(1)
  
  %try
  %  imread([basedir, datadir, imfiles(k,:)]);
  %catch
  %  ppm_weird=[ppm_weird; imfiles(k,:)];
  %  continue
  %end_try_catch
  
  imnum{length(imnum)+1}=rgb2gray(imread([basedir, datadir, imfiles(k,:)]));
  
endfor


