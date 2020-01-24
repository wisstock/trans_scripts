%%
% -------------------------CoNSTANTS---------------------------------------
%----Choise = 0 - if auto searching of maximum or else if manual(MidxA0,MidxB0) 
Choise = 0;
MidxA0 = 5;
MidxB0 = 5;
dMidx = 1;
%MaskMainA = 0 -if used maskA for translocations in B
MaskMainA = 1;


Name_dir = 'D:\Lab\Translocations_HPCA\Cell21\corr\proc'; %reading the contents of a directory
Name_MasterImg = 'Fluorescence 435nm_p';

MaskA = 'Fluorescence 435nm';
MaskB = 'Fluorescence  FRET';
MaskR = 'HFstim_505x_435y_';
clc
N_frm_base = 3;

dXYMask = 80;
aMask = 0.5;
 
dMax =20;

Name_dir_proc = '\proc4';
Name_seq_t = '_p';


N_frm_begin = 1;
amin = 1;
amax = 16000;
d = 1;
dT = 2;
D = 2;
g = 0.2;
h = ones(3,3)/9;%h = matrix, 3x3.(average value for 9 pixels  = filtration)
% surrounding pixels of some point from our image multiplied with this
% created matrix and divided by 9. 
%==========================================================================
%%
% -------------------------FILES-------------------------------------------
DirList = dir(Name_dir);
[Nfiles m] = size(DirList);
[m lMask] = size(MaskA); 

ind = 0; %
for i = 1:Nfiles 
    if DirList(i).isdir == 0 
       Name = DirList(i).name;
       if strcmp(Name(end-3:end),'.tif')&&strcmp(Name(1:lMask),MaskA)    
           ind = ind+1; 
           Count(ind) = i;
           if strcmp(Name(1:end-4),Name_MasterImg)
               Midx = ind; 
           end
       end
    end
end

mkdir([Name_dir Name_dir_proc]); 

NameTemp = DirList(Count(1)).name;
DirList(Count(1)).name = DirList(Count(Midx)).name;
DirList(Count(Midx)).name = NameTemp;
%==========================================================================
%%
%================================ MAIN PART ===============================
for fidx = 1:ind 
%%
% -------------------------PATH&FILES-MaskA--------------------------------
Name = DirList(Count(fidx)).name; 
Name = Name(1:end-4);

Name_seq = [Name_dir  '\'  Name  '.tif'];
Name_seq_temp1 = [Name_dir Name_dir_proc '\' Name Name_seq_t  '.tif'] ;
  
info = imfinfo(Name_seq);
[N_frm_end m] = size(info);
Delta = zeros(1,N_frm_end);
%==========================================================================
%%
% ----------------------IMG_BIG - MaskA------------------------------------
for idx = 1:N_frm_end
    Img_temp_org = imread(Name_seq,idx);
    Img_temp = mat2gray(Img_temp_org,[amin amax]); 
    if (idx == 1)
        Img_big_float = zeros(size(Img_temp));
    end 
    Img_big_float = Img_big_float + Img_temp;
end
Img_big_float = imfilter(Img_big_float/N_frm_end,h);
[Xval Yval] = size(Img_big_float);%1040õ1392
%==========================================================================
%%
% ------------------------BGR - MaskA -------------------------------------

   % ------------------------BGR - MaskA -------------------------------------

    T1 = 75/amax;
    BW3 = roicolor(Img_big_float,T1,1);
    BW3 = +BW3;
    S = sum(sum(BW3))/(Yval*Xval);
    
    while(S>1-g)
        T1 = T1+0.2/amax;
        BW3 = roicolor(Img_big_float,T1,1);
        BW3 = +BW3;
        S = sum(sum(BW3))/(Yval*Xval);
    end
    
    Bgr = sum(sum(Img_big_float.*(ones(size(BW3))-BW3)))/sum(sum(ones(size(BW3))-BW3))*amax;
%==========================================================================
%%
% ------------------------MASK - MaskA ------------------------------------

low = (Bgr + d)/amax;

       BW = roicolor(Img_big_float,low,1);   
       BW = +BW;
       
if fidx == 1
    Np0 = sum(sum(BW));
end
       Np_new = sum(sum(BW));
       
        if(Np_new>Np0)
             while(Np_new>Np0)
    
             low = low + 0.1/amax;
             BW = roicolor(Img_big_float,low,1);
             BW = +BW;
             Np_new = sum(sum(BW));
             end
        else
            while(Np0>Np_new)
              low = low  - 0.1/amax;
              BW = roicolor(Img_big_float,low,1);
              BW = +BW;
              Np_new = sum(sum(BW));
            end
        end
 %%       
 % ----------------------------Soma coord- All-----------------------------      
Img_big_float = (Img_big_float -Bgr/amax*ones(size(Img_big_float))).*BW;

if (fidx == 1)
SomaMAX=  max(max(Img_big_float));
BWSoma = roicolor(Img_big_float,aMask*SomaMAX,1);
BWSoma = +BWSoma;

XD = sum(BWSoma,2);

X0 = 1;
if XD(1) ==0
while(XD(X0)==0)
X0 = X0+1;
end
end

if X0 > dXYMask
    X0 = X0 - dXYMask;
else
    X0 = 1;
end


X1 =Xval;
if XD(Xval) ==0
while(XD(X1)==0)
X1 = X1-1;
end
end

if X1 < Xval - dXYMask
    X1 = X1 + dXYMask;
else
    X1 = Xval;
end


YD = sum(BWSoma,1);

Y0 = 1;

if YD(1)==0
while( YD(Y0)==0)
Y0 = Y0+1;
end
end

if Y0 > dXYMask
    Y0 = Y0 - dXYMask;
else
    Y0 = 1;
end


Y1 =Yval;
if YD(Yval)==0
while(YD(Y1)==0)
Y1 = Y1-1;
end
end

if Y1 < Yval - dXYMask
    Y1 = Y1 + dXYMask;
else
    Y1 = Yval;
end

end
%%
% ----------------------------fotobleaching comp -MaskA--------------------
for idx = N_frm_begin:N_frm_end
    Img_temp_org = imread(Name_seq,idx);
    Img_temp_float = mat2gray(Img_temp_org,[amin amax]);
%%
  
%%    
     if (idx == N_frm_begin)&&(fidx == 1)
        
        sum0 = sum(sum(Img_temp_float))/sum(sum(BW))*(2000/amax)/max(max(Img_temp_float));
     end
     
    sum1 = sum(sum(Img_temp_float))/sum(sum(BW));

    Img_temp_float = Img_temp_float*sum0/sum1; 
    %%
    if idx == N_frm_begin
        IMG_MAIN_A = Img_temp_float;
    end
   
Delta(idx) = sum(sum(abs(Img_temp_float - IMG_MAIN_A)))/sum(sum(IMG_MAIN_A ));
    %%
IMG_MAIN_A = Img_temp_float;    
    

end
%%
%-----------------------------------MaskA-SumTransl------------------------
%-----------------------------------ImgMain-MaskA--------------------------
for idx = 2:N_frm_base
    Img_temp_org = imread(Name_seq,idx);
    Img_temp_float = mat2gray(Img_temp_org,[amin amax]);
    if idx == 2
        IMG_MAIN_A = zeros(size(Img_temp_float));
    end   
        IMG_MAIN_A = IMG_MAIN_A + Img_temp_float;    
end
IMG_MAIN_A = IMG_MAIN_A/2;
%%
if Choise == 0
[Max MidxA] = max(Delta);
Img_temp_org = imread(Name_seq,MidxA);
IMG_MAX_A = mat2gray(Img_temp_org,[amin amax]);
else
    MidxA = MidxA0;
    for i = MidxA0:MidxA0+dMidx -1
        Img_temp_org = imread(Name_seq,MidxA);
        IMG_MAX_A_temp = mat2gray(Img_temp_org,[amin amax]);
        if i== MidxA0
            IMG_MAX_A = IMG_MAX_A_temp;
        else
            IMG_MAX_A = IMG_MAX_A_temp+IMG_MAX_A;
        end
    end
    IMG_MAX_A = IMG_MAX_A/dMidx;
    
end

IMG_DELTA_A = IMG_MAX_A - IMG_MAIN_A;
BW_A = roicolor(IMG_DELTA_A,dMax/amax,1);
BW_A = +BW_A;

if fidx == 1
    BW_AA = BW_A;
end


SumTranslA = zeros(N_frm_end -N_frm_begin+1,2);

if (fidx == 1)
%minValue = min(IMG_DELTA_A(:));
%maxValue = max(IMG_DELTA_A(:));
%imagesc(IMG_DELTA_A);
N = 256;
IMG_DELTA_A1 = IMG_DELTA_A - min(IMG_DELTA_A(:));
IMG_DELTA_A1 = (IMG_DELTA_A1/max(IMG_DELTA_A1(:)))*N;

IMG_DELTA_A0 = uint8(IMG_DELTA_A1);
IMG_DELTA_A0 = imadjust(IMG_DELTA_A0,[0.3 0.7]);

[M] = mode(IMG_DELTA_A0(:));
 
greenColorMap = [zeros(1, M + 2 ) linspace(0, 1, 256 - (M + 2))];
redColorMap = [linspace(1, 0, 256 - (256 - (M - 2))), zeros(1,256 - (M - 2))];
colorMap = [redColorMap; greenColorMap; zeros(1, 256)]';
%colormap(colorMap)
Name_seq_temp3 = [Name_dir Name_dir_proc '\' 'redgreen435' '.tif'] ;
imwrite(IMG_DELTA_A0, colorMap, Name_seq_temp3,'WriteMode','overwrite');

end

for idx = N_frm_begin:N_frm_end
    Img_temp_org = imread(Name_seq,idx);
    Img_temp_float = mat2gray(Img_temp_org,[amin amax]);
    
%% 
%---------------------------------SumTranslInSoma--------------------------
SBW = BW_AA(X0:X1,Y0:Y1);
IMG_D_A = Img_temp_float  - IMG_MAIN_A;
SumTranslA(idx,1) = sum(sum(IMG_D_A(X0:X1,Y0:Y1).*SBW))/sum(sum(IMG_MAIN_A(X0:X1,Y0:Y1).*SBW));
%%
%---------------------------------SumTranslInDendr-------------------------
SBW = BW_AA;
SBW(X0:X1,Y0:Y1) = zeros(size(SBW(X0:X1,Y0:Y1)));


SumTranslA(idx,2) = sum(sum(IMG_D_A.*SBW))/sum(sum(IMG_MAIN_A.*SBW));
end

xlswrite([Name_dir Name_dir_proc '\' MaskR  'sum_transl_example'  '.xlsx'],{'dendr' MaskA MaskB ' ' 'soma' MaskA MaskB },['HFs_' Name(lMask+1:end)],'A1');

xlswrite([Name_dir Name_dir_proc '\' MaskR  'sum_transl_example'  '.xlsx'],SumTranslA(:,2),['HFs_' Name(lMask+1:end)],'B2');
xlswrite([Name_dir Name_dir_proc '\' MaskR  'sum_transl_example'  '.xlsx'],SumTranslA(:,1),['HFs_' Name(lMask+1:end)],'F2');

%==========================================================================
%%


% -------------------------PATH&FILES-MaskB--------------------------------
Name = DirList(Count(fidx)).name;
Name = Name(1:end-4);

Name_seq = [Name_dir  '\'  MaskB Name(lMask+1:end)  '.tif'];
Name_seq_temp1 = [Name_dir Name_dir_proc '\' MaskB Name(lMask+1:end) Name_seq_t  '.tif'] ;    
    
info = imfinfo(Name_seq);
%[N_frm_end m] = size(info);

%==========================================================================
%%

%%
if fidx == 1
 Img_temp_org = imread(Name_seq,4);
 
Img_temp_org = localcontrast(Img_temp_org, 0.2, 0.8);
Img_temp = mat2gray(Img_temp_org,[amin amax]); 
Img_temp = imadjust(Img_temp);
level = adaptthresh(Img_temp,0.5);
BW0 = imbinarize(Img_temp,level);
BW0 = imclearborder(BW0,26);
%Img_temp2 = Img_temp.*BW0 ;
 %hl = fspecial('sobel');
 %Gy = imfilter(Img_temp2, hl);
 %Gx = imfilter(Img_temp2, hl');
 %G = (Gx.^2 + Gy.^2).^(1/1);
 %G2 = imadjust(G);
% G2 = imadjust(G2);
 %G3 = imadjust(G2);
 %A3 = im2bw(G3);
%B = ones(size(A3));
%A4 = B - A3;
%A5 = A4.*BW0;
 %A6 = imfill(A5);
%A8 = im2bw(A6);
se = strel('sphere',1);
BW0 = imdilate(BW0, se);

Name_seq_temp4 = [Name_dir Name_dir_proc '\' 'fret_mask' '.tif'] ;
imwrite(BW0, Name_seq_temp4,'WriteMode','overwrite');
end

BW_BB = BW0;

for idx = 2:N_frm_base
    Img_temp_org = imread(Name_seq,idx);
    Img_temp_float = mat2gray(Img_temp_org,[amin amax]);
    if idx == 2
        IMG_MAIN_B = zeros(size(Img_temp_float));
    end   
        IMG_MAIN_B = IMG_MAIN_B + Img_temp_float;    
end
IMG_MAIN_B = IMG_MAIN_B/2;
SumTranslB = [];

for idx = N_frm_begin:N_frm_end
    Img_temp_org = imread(Name_seq,idx);
    Img_temp_float = mat2gray(Img_temp_org,[amin amax]);


%%
%---------------------------------SumTranslInSoma--------------------------
SBW = BW_BB(X0:X1,Y0:Y1);
IMG_D_B = Img_temp_float  - IMG_MAIN_B;
SumTranslB(idx,1) = sum(sum(IMG_D_B(X0:X1,Y0:Y1).*SBW))/sum(sum(IMG_MAIN_B(X0:X1,Y0:Y1).*SBW));

%---------------------------------SumTranslInDendr-------------------------
SBW = BW_BB;
SBW(X0:X1,Y0:Y1) = zeros(size(SBW(X0:X1,Y0:Y1)));
SumTranslB(idx,2) = sum(sum(IMG_D_B.*SBW))/sum(sum(IMG_MAIN_B.*SBW));
end

xlswrite([Name_dir Name_dir_proc '\' MaskR  'sum_transl_example'  '.xlsx'],SumTranslB(:,2),['HFs_' Name(lMask+1:end)],'C2');
xlswrite([Name_dir Name_dir_proc '\' MaskR  'sum_transl_example'  '.xlsx'],SumTranslB(:,1),['HFs_' Name(lMask+1:end)],'G2');


%==========================================================================
end


