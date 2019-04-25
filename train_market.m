%这个文件是训练数据集的模型的
function train_id_net_vgg16(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('./url_data_market.mat') ;
imdb = imdb.imdb;
%imdb.images.set(1:10000) = 3;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_market_newattribute();
net.conserveMemory = true;
net.meta.normalization.averageImage = reshape([105.6920,99.1345,97.9152],1,1,3);
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 64;
opts.train.continue = true; 
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.expDir = './data/resnet_market';
opts.train.derOutputs = {'objective', 21.6,'objective_ubla',0.1,'objective_uw',0.1,'objective_ur',0.1,'objective_up',0.1,...
    'objective_uy',0.1,'objective_ugra',0.1,'objective_ublu',0.1,'objective_ugre',0.1,...
    'objective_dbla',0.1,'objective_dw',0.1,'objective_dpi',0.1,'objective_dp',0.1,...
    'objective_dy',0.1,'objective_dgra',0.1,'objective_dblu',0.1,'objective_dgre',0.1,'objective_dbro',0.1,...
    'objective_g',0.1,'objective_h',0.1,'objective_a',0.1,'objective_u',0.1,'objective_d',0.1,'objective_c',0.1,...
    'objective_bg1',0.1,'objective_ha',0.1,'objective_bg2',0.1,'objective_bg3',0.1} ;
%opts.train.gamma = 0.9;
opts.train.momentum = 0.9;
%opts.train.constraint = 5;
opts.train.learningRate = [0.1*ones(1,50),0.01*ones(1,5)] ;
opts.train.weightDecay = 0.0001;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;
%todo 注释addpath这一句
%addpath ../matlab_email
%todo 将不认识的这一句demo_email()注释掉
%demo_email();
% --------------------------------------------------------------------
function inputs = getBatch(imdb,batch,opts)
% --------------------------------------------------------------------
im_url = imdb.images.data(batch) ; 
im = vl_imreadjpeg(im_url,'Flip');
labels = imdb.images.label(batch) ;
labels_ubla = imdb.images.upblack(batch) ;
labels_uw = imdb.images.upwhite(batch) ;
labels_ur = imdb.images.upred(batch) ;
labels_up = imdb.images.uppurple(batch) ;
labels_uy = imdb.images.upyellow(batch) ;
labels_ugra = imdb.images.upgray(batch) ;
labels_ublu = imdb.images.upblue(batch) ;
labels_ugre = imdb.images.upgreen(batch) ;
labels_dbla = imdb.images.downblack(batch) ;
labels_dw = imdb.images.downwhite(batch) ;
labels_dpi = imdb.images.downpink(batch) ;
labels_dp = imdb.images.downpurple(batch) ;
labels_dy = imdb.images.downyellow(batch) ;
labels_dgra = imdb.images.downgray(batch) ;
labels_dblu = imdb.images.downblue(batch) ;
labels_dgre = imdb.images.downgreen(batch) ;
labels_dbro = imdb.images.downbrown(batch); 

labels_g = imdb.images.gender(batch) ;
labels_h = imdb.images.hair(batch) ;
labels_u = imdb.images.up(batch) ;
labels_d = imdb.images.down(batch) ;
labels_c = imdb.images.clothes(batch) ;
labels_ha = imdb.images.hat(batch) ;
labels_a = imdb.images.age(batch) ;
labels_bg1 = imdb.images.backpack(batch) ;
labels_bg2 = imdb.images.bag(batch) ;
labels_bg3 = imdb.images.handbag(batch) ;

batchsize = numel(im);
oim = zeros(224,224,3,batchsize,'single');
for i=1:batchsize
    x = randi(33);
    y = randi(33);
    imt = im{i};
    temp = imt(x:x+223,y:y+223,:);
    oim(:,:,:,i) = temp;
    r = rand>0.5;
    if r 
        oim(:,:,:,i) = temp;
    else oim(:,:,:,i) = fliplr(temp);
    end
end
oim = bsxfun(@minus,oim,opts.averageImage); 
inputs = {'data',gpuArray(oim),'label',labels,'label2',labels_ubla,'label3',labels_uw,'label4',labels_ur,...
    'label5',labels_up,'label6',labels_uy,'label7',labels_ugra,'label8',labels_ublu,'label9',labels_ugre,...
    'label10',labels_dbla,'label11',labels_dw,'label12',labels_dpi,'label13',labels_dp,...
    'label14',labels_dy,'label15',labels_dgra,'label16',labels_dblu,'label17',labels_dgre,'label18',labels_dbro,...
    'label19',labels_g,'label20',labels_h,'label21',labels_a,'label22',labels_u,'label23',labels_d,'label24',labels_c,...
    'label25',labels_bg1,'label26',labels_ha,'label27',labels_bg2,'label28',labels_bg3};
