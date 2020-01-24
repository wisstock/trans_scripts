% background correction
% constants
Name_dir = 'D:\Lab\Translocations_HPCA\Cell21\corr';
Name_dir_proc = '\proc'; % directrotu for processed files
Name_seq_t = '_p'; % name of processed file = it's name + _p
DirList = dir(Name_dir);
[Nfiles, ~] = size(DirList);
N_frm_begin = 1;
amin = 0;
amax = 17000;
d = 25;
g = 0.2;
h = ones(3,3)/9;
MaskA = 'Fluorescence 435nm';% file with this name is recognized as MaskA
MaskB = 'Fluorescence  FRET';% file with this name is recognized as MaskA
Name_MasterImg = 'Fluorescence 435nm';


mkdir([Name_dir Name_dir_proc]); % creation of the directory 
[~, lMask] = size(MaskA); 
ind = 0;
%------counting of files------------------------------------

for i = 1:Nfiles 
    if DirList(i).isdir == 0 
       Name = DirList(i).name;
       if strcmp(Name(end-3:end),'.tif') && strcmp(Name(1:lMask),MaskA)    
           ind = ind+1; 
           Count(ind) = i;
           if strcmp(Name(1:end-4),Name_MasterImg)
               Midx = ind; 
           end
       end
    end
end

NameTemp = DirList(Count(1)).name;
DirList(Count(1)).name = DirList(Count(Midx)).name;
DirList(Count(Midx)).name = NameTemp;
%--------------main part----------------------------------
for fidx = 1:ind
    Name = DirList(Count(fidx)).name; 
    Name = Name(1:end-4); 
    Name505 = Name;
    Name505(1:18) = MaskB;
    Names = [];
    Names.Name_seq(1,:) = [Name_dir '\'  Name  '.tif'];
    Names.Name_seq(2,:) = [Name_dir '\'  Name505  '.tif'];


    Names.Name_seq_temp(1,:)  = [Name_dir Name_dir_proc '\' Name   '.tif'] ;
    Names.Name_seq_temp(2,:) = [Name_dir Name_dir_proc '\' Name505   '.tif'] ;

    Names.Name_seq_p(1,:)  = [Name_dir Name_dir_proc '\' Name  Name_seq_t '.tif'] ;
    Names.Name_seq_p(2,:) = [Name_dir Name_dir_proc '\' Name505 Name_seq_t  '.tif'] ;


    info = imfinfo(Names.Name_seq(1,:));%counting files 
    [N_frm_end, ~] = size(info);

    numChannels = 2;
    for ii = 1:numChannels
        for id = 1:N_frm_end
            Img_temp_org = imread(Names.Name_seq(ii,:), id);
            Img_temp = mat2gray(Img_temp_org, [amin amax]);
            if (id == 1)
                Img_big_float = zeros(size(Img_temp)); 
            end
            Img_big_float = Img_big_float + Img_temp;
        end
        Img_big_float = imfilter(Img_big_float/N_frm_end,h);
        [Xval, Yval] = size(Img_big_float);%1040x1392        
% ------------------------BGR correction ------------------------------------
        T1 = 250/amax;% Online aquisition adds 250 units of intensity to each pixel
        BW3 = roicolor(Img_big_float,T1,1); % mask for non-background pixels 
        S = sum(sum(BW3))/(Yval*Xval); %calculating of percentage of non-bgr pixels
        %it needs to be 80% if no- the threshold will be higher 
        while(S > 1-g)
            T1 = T1+0.2/amax;
            BW3 = roicolor(Img_big_float,T1,1);
            BW3 = +BW3;
            S = sum(sum(BW3))/(Yval*Xval);
        end
        Bgr = sum(sum(Img_big_float.*(ones(size(BW3))-BW3)))/sum(sum(ones(size(BW3))-BW3))*amax;
        %defining and adjusting bgr mask for each frame 
        low = (Bgr+d)/amax;
        BW = roicolor(Img_big_float,low,1);
        BW = +BW;
        if fidx == 1
            Np0 = sum(sum(BW));  
        end
        Np_new = sum(sum(BW));
        if(Np_new>Np0)
            while (Np_new>Np0)
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
        for idd = N_frm_begin:N_frm_end
            Img_temp_org = imread(Names.Name_seq(ii,:),idd);
            Img_temp_float = mat2gray(Img_temp_org,[amin amax]);
            %%bgr correction 
            Img_temp_float = Img_temp_float - ones(size(Img_temp_float))*Bgr/amax;
            Img_temp_float = imfilter(Img_temp_float,h);
            Img_temp_float = Img_temp_float.*BW;
            [Img_temp_org,map] = gray2ind(Img_temp_float,amax);
            %saving
            if (idd == N_frm_begin)
                imwrite(Img_temp_org,Names.Name_seq_p(ii,:),'WriteMode','overwrite');
            else
                imwrite(Img_temp_org,Names.Name_seq_p(ii,:),'WriteMode','append');
            end         
        end
    end 
end
