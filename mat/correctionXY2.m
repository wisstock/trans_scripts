Name_dir2 = 'D:\Lab\Translocations_HPCA\Cell21';
DirList = dir(Name_dir2);
Name_dir_proc = '\corr';
Name_seq_t = '\p';
amin = 0;
amax = 17000;
MaskA = 'Fluorescence 435nm';
Mask_MasterImg = 'Fluorescence 435nm';

mkdir([Name_dir2 Name_dir_proc]); %folder for processed images
[Nfiles, ~] = size(DirList);
[~, lMask] = size(MaskA);
ind = 0;


% This loop gets the indices of the .tif files and index of Master
% image file
for i = 1:Nfiles
    if DirList(i).isdir == 0
        Name = DirList(i).name;
        if strcmp(Name(end-3:end), '.tif') && strcmp(Name(1:lMask), MaskA)
            ind = ind + 1;
            Count(ind) = i;
            if strcmp(Name(1:end-4), Mask_MasterImg)
                Midx = ind;
            end
        end        
    end
end


NameTemp = DirList(Count(1)).name;
DirList(Count(1)).name = DirList(Count(Midx)).name;
DirList(Count(Midx)).name = NameTemp;


for fidx = 1:ind
    Name = DirList(Count(fidx)).name; 
    Name = Name(1:end-4); 
    Name505 = Name;
    Name505(1:18) = 'Fluorescence  FRET';
    Names = [];
    Names.Name_seq(1,:) = [Name_dir2 '\'  Name '.tif'];
    %path for accessing 
    Names.Name_seq_temp(1,:)  = [Name_dir2 Name_dir_proc '\' Name  '.tif'] ;
    Names.Name_seq_temp(2,:) = [Name_dir2 Name_dir_proc '\' Name505  '.tif'] ;
    
    % accesing frames of a .tif file
    info = imfinfo(Names.Name_seq(1,:));
    [N_frm_end, ~] = size(info);
    N_frm_begin = 1; 
    
    for idx = N_frm_begin:2:N_frm_end
        %converting first frm to 1/0 grayscale img
        Img1 = imread(Names.Name_seq(1,:), idx);
        Img11 = mat2gray(Img1,[amin amax]);
        %converting second frame to 1/0 grayscale img
        idxx=idx+1;
        Img2 = imread(Names.Name_seq(1,:), idxx);
        Img21 = mat2gray(Img2,[amin amax]);
        % prior normalization for better correlation
        max1 = max(max(Img11));
        max2 = max(max(Img21));
        
        norm1 = Img11/max1;
        norm2 = Img21/max2;
        
        norm1 = norm1 - 0.018;
        norm2 = norm2 - 0.018;
        
        % increases the contrast of imgs
        norm1 = imadjust(norm1);
        norm2 = imadjust(norm2);
        
        % correction
        if (idx == N_frm_begin)
            t2 = autoXY_shift_new(norm1,norm2, 'fun', 'nonreflectivesimilarity');
            Rfixed = imref2d(size(Img1));
            T2 = imwarp(Img1, t2, 'linear','OutputView', Rfixed);
        else    
            T2 = imwarp(Img1, t2, 'linear','OutputView', Rfixed);
        end 
        
        % saving the images 
        if (idx == N_frm_begin)
            imwrite(T2, Names.Name_seq_temp(1,:), 'WriteMode', 'overwrite');
        else
            imwrite(T2, Names.Name_seq_temp(1,:), 'WriteMode', 'append');
        end
        
        
        if (idx == N_frm_begin)
            imwrite(Img2, Names.Name_seq_temp(2,:), 'WriteMode', 'overwrite');
        else
            imwrite(Img2, Names.Name_seq_temp(2,:), 'WriteMode', 'append');
        end
    end
end    

%%% What imref2d does?
%%% Why do we subtract 0.018 from normalized imgs?
%%% Why do we apply imwarp only to 1 of the 2 frame and processed imgs for FRET are just 'Img2' without correction? 