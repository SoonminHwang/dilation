addpath('data/pascal_voc/VOCcode');

VOCinit;

% VOCopts.datadir = fullfile('data', 'pascal_voc', 'VOC2012');
VOCopts.resdir = fullfile('jobs', 'pascal_voc', 'joint', 'results', 'epoch_04');
VOCopts.seg.clsrespath=[VOCopts.resdir '/%s.png'];

% GT image path
VOCopts.seg.clsimgpath=[VOCopts.datadir VOCopts.dataset '/SegmentationClass/%s.png'];

[ acc, macc, conf, rawcounts ] = VOCevalseg(VOCopts);