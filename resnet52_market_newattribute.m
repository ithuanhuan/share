function net = resnet52_market_attributecolor_new()
netStruct = load('./data/imagenet-resnet-50-dag.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
net.removeLayer('fc1000');
net.removeLayer('prob');
for i = 1:numel(net.params)
    if(mod(i,2)==0)
        net.params(i).learningRate=0.02;
    else net.params(i).learningRate=0.001;
    end
    net.params(i).weightDecay=1;
end
dropoutBlock = dagnn.DropOut('rate',0.9);
net.addLayer('dropout',dropoutBlock,{'pool5'},{'pool5d'},{});
%-----reID
fc751Block = dagnn.Conv('size',[1 1 2048 751],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc751',fc751Block,{'pool5d'},{'prediction'},{'fc751f','fc751b'});
net.addLayer('softmaxloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction','label'},'objective');
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction','label'}, 'top5err') ;

%----upblack
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('upblack',cameraBlock,{'pool5'},{'prediction_ubla'},{'black','notblack'});
net.addLayer('upblackloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_ubla','label2'},'objective_ubla');
net.addLayer('top1err_upblack', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_ubla','label2'}, 'top1err_upblack') ;
%----upwhite
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('upwhite',cameraBlock,{'pool5'},{'prediction_uw'},{'white','notwhite'});
net.addLayer('upwhiteloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_uw','label3'},'objective_uw');
net.addLayer('top1err_upwhite', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_uw','label3'}, 'top1err_upwhite') ;
%----upred
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('upred',cameraBlock,{'pool5'},{'prediction_ur'},{'red','notred'});
net.addLayer('upredloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_ur','label4'},'objective_ur');
net.addLayer('top1err_upred', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_ur','label4'}, 'top1err_upred') ;
%----uppurple
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('uppurple',cameraBlock,{'pool5'},{'prediction_up'},{'purple','notpurple'});
net.addLayer('uppurpleloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_up','label5'},'objective_up');
net.addLayer('top1err_uppurple', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_up','label5'}, 'top1err_uppurple') ;
%----upyellow
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('upyellow',cameraBlock,{'pool5'},{'prediction_uy'},{'yellow','notyellow'});
net.addLayer('upyellowloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_uy','label6'},'objective_uy');
net.addLayer('top1err_upyellow', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_uy','label6'}, 'top1err_upyellow') ;
%----upgray
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('upgray',cameraBlock,{'pool5'},{'prediction_ugra'},{'gray','notgray'});
net.addLayer('upgrayloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_ugra','label7'},'objective_ugra');
net.addLayer('top1err_upgray', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_ugra','label7'}, 'top1err_upgray') ;
%----upblue
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('upblue',cameraBlock,{'pool5'},{'prediction_ublu'},{'blue','notblue'});
net.addLayer('upblueloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_ublu','label8'},'objective_ublu');
net.addLayer('top1err_upblue', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_ublu','label8'}, 'top1err_upblue') ;
%----upgreen
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('upgreen',cameraBlock,{'pool5'},{'prediction_ugre'},{'green','notgreen'});
net.addLayer('upgreenloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_ugre','label9'},'objective_ugre');
net.addLayer('top1err_upgreen', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_ugre','label9'}, 'top1err_upgreen') ;

%----downblack
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('downblack',cameraBlock,{'pool5'},{'prediction_dbla'},{'black1','notblack1'});
net.addLayer('downblackloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_dbla','label10'},'objective_dbla');
net.addLayer('top1err_downblack', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_dbla','label10'}, 'top1err_downblack') ;
%----downwhite
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('downwhite',cameraBlock,{'pool5'},{'prediction_dw'},{'white1','notwhite1'});
net.addLayer('downwhiteloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_dw','label11'},'objective_dw');
net.addLayer('top1err_downwhite', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_dw','label11'}, 'top1err_downwhite') ;
%----downpink
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('downpink',cameraBlock,{'pool5'},{'prediction_dpi'},{'pink','notpink'});
net.addLayer('downpinkloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_dpi','label12'},'objective_dpi');
net.addLayer('top1err_downpink', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_dpi','label12'}, 'top1err_downpink') ;
%----downpurple
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('downpurple',cameraBlock,{'pool5'},{'prediction_dp'},{'purple1','notpurple1'});
net.addLayer('downpurpleloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_dp','label13'},'objective_dp');
net.addLayer('top1err_downpurple', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_dp','label13'}, 'top1err_downpurple') ;
%----downyellow
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('downyellow',cameraBlock,{'pool5'},{'prediction_dy'},{'yellow1','notyellow1'});
net.addLayer('downyellowloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_dy','label14'},'objective_dy');
net.addLayer('top1err_downyellow', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_dy','label14'}, 'top1err_downyellow') ;
%----downgray
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('downgray',cameraBlock,{'pool5'},{'prediction_dgra'},{'gray1','notgray1'});
net.addLayer('downgrayloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_dgra','label15'},'objective_dgra');
net.addLayer('top1err_downgray', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_dgra','label15'}, 'top1err_downgray') ;
%----downblue
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('downblue',cameraBlock,{'pool5'},{'prediction_dblu'},{'blue1','notblue1'});
net.addLayer('downblueloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_dblu','label16'},'objective_dblu');
net.addLayer('top1err_downblue', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_dblu','label16'}, 'top1err_downblue') ;
%----downgreen
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('downgreen',cameraBlock,{'pool5'},{'prediction_dgre'},{'green1','notgreen1'});
net.addLayer('downgreenloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_dgre','label17'},'objective_dgre');
net.addLayer('top1err_downgreen', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_dgre','label17'}, 'top1err_downgreen') ;
%----downbrown
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('downbrown',cameraBlock,{'pool5'},{'prediction_dbro'},{'brown','notbrown'});
net.addLayer('downbrownloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_dbro','label18'},'objective_dbro');
net.addLayer('top1err_downbrown', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_dbro','label18'}, 'top1err_downbrown') ;

%----gender
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('gender',cameraBlock,{'pool5'},{'prediction_g'},{'genderf','genderb'});
net.addLayer('genderloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_g','label19'},'objective_g');
net.addLayer('top1err_gender', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_g','label19'}, 'top1err_gender') ;
%----hair
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('hair',cameraBlock,{'pool5'},{'prediction_h'},{'hairs','hairl'});
net.addLayer('hairloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_h','label20'},'objective_h');
net.addLayer('top1err_hair', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_h','label20'}, 'top1err_hair') ;
% %----age
cameraBlock = dagnn.Conv('size',[1 1 2048 4],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('age',cameraBlock,{'pool5'},{'prediction_a'},{'age1','age2'});
net.addLayer('ageloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_a','label21'},'objective_a');
net.addLayer('top1err_age', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_a','label21'}, 'top1err_age') ; 
%----up
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('up',cameraBlock,{'pool5'},{'prediction_u'},{'upl','ups'});
net.addLayer('uploss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_u','label22'},'objective_u');
net.addLayer('top1err_up', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_u','label22'}, 'top1err_up') ;
%----down
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('down',cameraBlock,{'pool5'},{'prediction_d'},{'downl','downs'});
net.addLayer('downloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_d','label23'},'objective_d');
net.addLayer('top1err_down', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_d','label23'}, 'top1err_down') ;
%----clothes
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('clothes',cameraBlock,{'pool5'},{'prediction_c'},{'clothess','clothesp'});
net.addLayer('clothesloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_c','label24'},'objective_c');
net.addLayer('top1err_clothes', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_c','label24'}, 'top1err_clothes') ;
%----bagtype
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('bag1',cameraBlock,{'pool5'},{'prediction_bg1'},{'bag11','bag12'});
net.addLayer('bag1loss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_bg1','label25'},'objective_bg1');
net.addLayer('top1err_bag1', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_bg1','label25'}, 'top1err_bag1') ; 
%----hat
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('hat',cameraBlock,{'pool5'},{'prediction_ha'},{'hatnone','hathave'});
net.addLayer('hatloss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_ha','label26'},'objective_ha');
net.addLayer('top1err_hat', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_ha','label26'}, 'top1err_hat') ; 
%----bagtype
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('bag2',cameraBlock,{'pool5'},{'prediction_bg2'},{'bag21','bag22'});
net.addLayer('bag2loss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_bg2','label27'},'objective_bg2');
net.addLayer('top1err_bag2', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_bg2','label27'}, 'top1err_bag2') ;
%----bagtype
cameraBlock = dagnn.Conv('size',[1 1 2048 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('bag3',cameraBlock,{'pool5'},{'prediction_bg3'},{'bag31','bag32'});
net.addLayer('bag3loss',dagnn.Loss('loss', 'softmaxlog'),{'prediction_bg3','label28'},'objective_bg3');
net.addLayer('top1err_bag3', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_bg3','label28'}, 'top1err_bag3') ;
net.initParams();
end

