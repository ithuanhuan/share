clear;
p = '/Users/huan/Downloads/re-id/dataset/Market1501/bounding_box_train/';
pp = '/Users/huan/Downloads/re-id/dataset/Market1501/bounding_box_train/*.jpg';
pw = '/Users/huan/Downloads/re-id/dataset/Market1501/bounding_box_train_256/';
load('market_attribute.mat');
upblack = market_attribute.train.upblack;
upwhite = market_attribute.train.upwhite;
upred = market_attribute.train.upred;
uppurple = market_attribute.train.uppurple;
upyellow = market_attribute.train.upyellow;
upgray = market_attribute.train.upgray;
upblue = market_attribute.train.upblue;
upgreen = market_attribute.train.upgreen;
downblack = market_attribute.train.downblack;
downwhite = market_attribute.train.downwhite;
downpink = market_attribute.train.downpink;
downpurple = market_attribute.train.downpurple;
downyellow = market_attribute.train.downyellow;
downgray = market_attribute.train.downgray;
downblue = market_attribute.train.downblue;
downgreen = market_attribute.train.downgreen;
downbrown = market_attribute.train.downbrown;
backpack = market_attribute.train.backpack;
handbag = market_attribute.train.handbag;
bag = market_attribute.train.bag;
gender = market_attribute.train.gender;
up = market_attribute.train.up;
down = market_attribute.train.down;
clothes = market_attribute.train.clothes;
hat = market_attribute.train.hat;
age = market_attribute.train.age;
hair = market_attribute.train.hair;
imdb.meta.sets=['train','test'];
file = dir(pp);
counter_data=1;
counter_last = 1;
class = 0;
c_last = '';
cc = [];
m = zeros(1,1,3);
for i=1:length(file)
    url = strcat(p,file(i).name);
    im = imread(url);
    im = imresize(im,[256,256]); % save 256*256 image in advance for faster IO
    m = m+mean(mean(im),2);
    url256 = strcat(pw,file(i).name);
    imwrite(im,url256);
    c = strsplit(file(i).name,'_');
    if(~isequal(c{1},c_last))
        class = class + 1;
        %if class == 652 
        %    break; 
        % end
        fprintf('%d::%d\n',class,counter_data-counter_last);
        cc=[cc;counter_data-counter_last];
        c_last = c{1};
        counter_last = counter_data;
    end
    
    imdb.images.data(counter_data) = cellstr(url256); 
    imdb.images.label(:,counter_data) = class;
    imdb.images.upblack(:,counter_data) = upblack(class);
    imdb.images.upwhite(:,counter_data) = upwhite(class);
    imdb.images.upred(:,counter_data) = upred(class);
    imdb.images.uppurple(:,counter_data) = uppurple(class);
    imdb.images.upyellow(:,counter_data) = upyellow(class);
    imdb.images.upblue(:,counter_data) = upblue(class);
    imdb.images.upgreen(:,counter_data) = upgreen(class);
    imdb.images.upgray(:,counter_data) = upgray(class);
    
    imdb.images.downblack(:,counter_data) = downblack(class);
    imdb.images.downwhite(:,counter_data) = downwhite(class);
    imdb.images.downpink(:,counter_data) = downpink(class);
    imdb.images.downpurple(:,counter_data) = downpurple(class);
    imdb.images.downyellow(:,counter_data) = downyellow(class);
    imdb.images.downblue(:,counter_data) = downblue(class);
    imdb.images.downgreen(:,counter_data) = downgreen(class);
    imdb.images.downgray(:,counter_data) = downgray(class);
    imdb.images.downbrown(:,counter_data) = downbrown(class);
    
    imdb.images.clothes(:,counter_data) = clothes(class);
    imdb.images.gender(:,counter_data) = gender(class);
    imdb.images.up(:,counter_data) = up(class);
    imdb.images.down(:,counter_data) = down(class);
    imdb.images.hat(:,counter_data) = hat(class);
    imdb.images.backpack(:,counter_data) = backpack(class);
    imdb.images.handbag(:,counter_data) = handbag(class);
    imdb.images.bag(:,counter_data) = bag(class);
    imdb.images.age(:,counter_data) = age(class);
    imdb.images.hair(:,counter_data) = hair(class);
    counter_data = counter_data + 1;
end

m = m/(i-1);
disp(m);
s = counter_data-1;
imdb.images.set = ones(1,s);
imdb.images.set(:,randi(s,[round(0.1*s),1])) = 2;

% no validation for small class 
cc = [cc;9];
cc = cc(2:end);
list = find(imdb.images.set==2);
for i=1:numel(list)
    if cc(imdb.images.label(list(i)))<10
        imdb.images.set(i)=1;
    end
end
%save('url_data_market_651','imdb','-v7.3');
save('url_data_market.mat','imdb','-v7.3');
