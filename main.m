std_rgb= imread('/Users/apple/Desktop/Progarm/matlab/Coloredphoto/test/standard.png');
stdImg= ~getBW_(std_rgb);
figure();
imshow(stdImg);
title('标准图');
orgin_src_rgb=imread('/Users/apple/Desktop/Progarm/matlab/Coloredphoto/test/multidefect1.jpg');
src_rgb = imread('/Users/apple/Desktop/Progarm/matlab/Coloredphoto/test/multidefect.png');
srcImg= ~getBW_(src_rgb);
figure();
imshow(srcImg);
title('待检测图');

% A=roateImg(stdImg,srcImg);
% se=strel('square',1);
% A_=imopen(A,se);
% figure()
% imshow(A_);
% title('匹配后的待检测pcb图像');

p_=xor(stdImg,srcImg);
figure()
imshow(p_);
title('异或图');
p_=medfilt2(p_,[7 7]);  %3
MN=[11 11]; %2
se=strel('rectangle',MN);%定义结构元素
%p__=imerode(p_,se);%腐蚀运算
p=imdilate(p_,se);%膨胀运算
[B,L] = bwboundaries(p);    %返回二值图像的边界 B是胞元数组（每个胞元里面是一簇的数据点） L是二维的标签数组
figure,imshow(p);  %形态学处理后的异或图
hold on;

for k = 1:length(B)
    boundary = B{k};
    plot(boundary(:,2),boundary(:,1),'g','LineWidth',2);       %画出边界图像
end

[L,N] = bwlabel(p);        %返回一个和p大小相同的L矩阵，包含了标记了p中每个连通区域的类别标签，N是连通区域的个数
img_rgb = label2rgb(L,'hsv',[.5 .5 .5],'shuffle'); %根据L的数值对应，默认对应到colormap(jet)的色彩，返回RGB矩阵
figure,imshow(img_rgb);
hold on
for k =1:length(B)
    boundary = B{k};
    plot(boundary(:,2),boundary(:,1),'w','LineWidth',2);
    text(boundary(1,2)-11,boundary(1,1)+11,num2str(k),'Color','y','Fontsize',14,'FontWeight','bold');   %带数字标记的连通域
end

stats = regionprops(L,'all');   %统计的属性值保留在stats内
temp = zeros(1,N);
for k = 1:N
    %计算thinness ratio（细度比例）
    temp(k) = 4 * pi * stats(k,1).Area / (stats(k,1).Perimeter)^2;
    stats(k,1).ThinnessRatio = temp(k);
    %计算aspect ratio
    temp(k) = (stats(k,1).BoundingBox(3))/(stats(k,1).BoundingBox(4));
    stats(k,1).AspectRatio = temp(k);
end

areas = zeros(1,N);
for k = 1:N
    areas(k) = stats(k).Area;
end
areas_=sort(areas);

rects= zeros(4,N);
std_Feats= zeros(3,N);
src_Feats= zeros(3,N);
error=zeros(3,N);
labels= zeros(1,N);
new_folder='/Users/apple/Desktop/Progarm/matlab/Grayphoto/img';
mkdir(new_folder);
for k= 1:N
    for j= 1:4
        rects(j,k)=stats(k).BoundingBox(j);
    end
    pic=~srcImg(rects(2,k)-5:rects(2,k)+rects(4,k)+5,rects(1,k)-5:rects(1,k)+rects(3,k)+5);
    figure()
    imshow(pic);
%     imwrite(pic,'/Users/apple/Desktop/Progarm/matlab/Grayphoto/img/k.jpg');
    imwrite(pic,strcat('/Users/apple/Desktop/Progarm/matlab/Grayphoto/img/',num2str(k),'.','jpg'));
    std_Feats(1,k)=bwarea(~stdImg(rects(2,k):rects(2,k)+rects(4,k),rects(1,k):rects(1,k)+rects(3,k)));
    src_Feats(1,k)=bwarea(~srcImg(rects(2,k):rects(2,k)+rects(4,k),rects(1,k):rects(1,k)+rects(3,k)));
    [L1,num1] = bwlabel(~stdImg(rects(2,k)-2:rects(2,k)+rects(4,k)+2,rects(1,k)-2:rects(1,k)+rects(3,k)+2));
    std_Feats(2,k)=num1;
    [L2,num2]=bwlabel(~srcImg(rects(2,k)-2:rects(2,k)+rects(4,k)+2,rects(1,k)-2:rects(1,k)+rects(3,k)+2));
    src_Feats(2,k)=num2;
    std_Feats(3,k)=bweuler(~stdImg(rects(2,k)-2:rects(2,k)+rects(4,k)+2,rects(1,k)-2:rects(1,k)+rects(3,k)+2));
    src_Feats(3,k)=bweuler(~srcImg(rects(2,k)-2:rects(2,k)+rects(4,k)+2,rects(1,k)-2:rects(1,k)+rects(3,k)+2));
    for i=1:3
        error(i,k)=src_Feats(i,k)-std_Feats(i,k);
    end
    if error(1,k)>0
       if error(2,k)>0
            if error(3,k)==0
               labels(k)=5;
            end
       elseif error(2,k)==0
           if error(3,k)>0
              labels(k)=6;
           elseif error(3,k)==0
              labels(k)=3;
           end
       else
           if error(2,k)<0
              labels(k)=2; 
           end
       end
    else
        if error(2,k)>0
            if error(3,k)>0
               labels(k)=1;
            end
        else error(2,k)==0
            if  error(3,k)==0
                labels(k)=7;
            elseif error(3,k)<0
                  labels(k)=4;
            end
        end
    end
end

figure();
imshow(srcImg);   %
title('待检测图缺陷分布');
for i=1:N
   rectangle( 'Position', [rects(1,i) rects(2,i) rects(3,i) rects(4,i)], 'EdgeColor', 'r');  % pos为矩形框位置
   switch labels(i)
            case 1 
           text(rects(1,i)+rects(3,i),rects(2,i)+rects(4,i), '开路','FontSize',20,'Color','r');
            case 2 
           text(rects(1,i)+rects(3,i),rects(2,i)+rects(4,i), '短路','FontSize',20,'Color','r');
            case 3 
           text(rects(1,i)+rects(3,i),rects(2,i)+rects(4,i), '毛刺','FontSize',20,'Color','r');
            case 4 
           text(rects(1,i)+rects(3,i),rects(2,i)+rects(4,i), '空洞','FontSize',20,'Color','r');
            case 5 
           text(rects(1,i)+rects(3,i),rects(2,i)+rects(4,i), '针孔','FontSize',20,'Color','r');
            case 6 
           text(rects(1,i)+rects(3,i),rects(2,i)+rects(4,i), '缺孔','FontSize',20,'Color','r');
            otherwise
           text(rects(1,i)+rects(3,i),rects(2,i)+rects(4,i), '缺损','FontSize',20,'Color','r');
   end
end
