function tform=autoXY_shift_new(C0, C1, varargin)
% automatically creates trasformation structure for use with imtransform
% function
%
% tform=autoXY_shift(Image1,Image2)
% tform=autoXY_shift(Image1,Image2,fun)
% tform=autoXY_shift(Image1,Image2,Param1,Val1,Param2,Val2...)
%
%   tform=autoXY_shift(Image1,Image2) creates tform transformation
%   structure with default settings for Image1 to maximum fit Image2 using
%   maximum correlation.
%   
%   tform=autoXY_shift(Image1,Image2,fun) is a compatible syntax mode for
%   scrypts written using version 2.1. Runs with default settings, except
%   a transformation type needs to be set by fun
%   
%   tform=autoXY_shift(Image1,Image2,Param1,Val1,Param2,Val2...) arguments
%   after Image1 and Image2 are set in style 'property', value.
%
%       Complete list of properties:
%           - 'fun' - a transformation function type. Allowed values are: 
%               'nonreflective similarity','similarity','affine',
%               'projective','polynomial','piecewise linear','lwm'. The
%               default value is 'nonreflective similarity'.
%           - 'bin' - binning factor for first stage raw shift.
%               Please, note that correction can only be done when not more
%               then 4 pixel shift is present. The images are first
%               downsampled by binning factor, the raw correlation grid is
%               created, then the pre-corrected grid is used to finecorrect
%               the original sized image. The default value for 'bin' is 4.
%           - 'threshold' - sets the threshold for grid points to be
%               discarded. (max - min) * threshold + min formula is applied.
%           - 'grid' - sets the density of the grid in pixels between each
%               other. Default is 5. Increase this to speed up (precision 
%               may be affected as well).
%           - 'border' - sets border to avoid artefacts of missalignment on
%               edges of image. Default is 5 (value in pixels)
%       Tips:
%           - to speed up processing of high resolution images use higher
%           values of 'bin' (this will make faster pre-shift correction)
%           - on high quality aligned images one can try setting 'bin'to 2
%           and setting grid to 10 or more (depending on image resolution)
%           - averaged images instead of single would be best choice
%           - lowpass filtering may be necesary for high-noise images
%           - if something goes wrong try default settings
%   Example1:
%       tform=autoXY_shift(C0, C1);
%       C0_corrected = imtransform(C0, tform, 'bicubic','XData',[1 size(C0,2)],'YData',[1 size(C0,1)]);
%
%   Example2:
%       tform=autoXY_shift(C0, C1, 'bin', 4, 'grid', 5, 'fun', 'affine', 'threshold', 0.01);
%       C0_corrected = imtransform(C0, tform, 'bicubic','XData',[1 size(C0,2)],'YData',[1 size(C0,1)]);
%
%   Example3:
%       tform=autoXY_shift_new(C0, C1, 'bin', 2, 'grid', 5, 'fun', 'polynomial','degree',3, 'threshold', 0.001);
%       Rfixed = imref2d(size(C0)); 
%       C0c = imwarp(C0, tform, 'linear','OutputView', Rfixed);
%
%   See also IMTRANSFORM, CPCORR, CP2TFORM
%
%   Version history:
%       not fixed bug in version 3.0 with tform.tdata.T to be mixed with tform.tdata.Tinv
%       fixed in 3.2
%       udpated tform structure, imtransform -> imwarp
%
%   Copyright Volodymyr Cherkas
%   $Revision: 3.3 $  $Date: 2015-Nov-20 13:50 $

switch nargin
    case {0 1}
        error('Error, at least 2 variables allowed to create tform!')
    case 2 % autoXY_shift with default settings
        set.fun='nonreflectivesimilarity';
        set.bin=4;          % default bin for preshift (test if works with even)
        set.grid_step=5;    % cpcorr works well within +/-5pixels, thus grid <5 is not recommended
        set.threshold=0.02; % threshold as a relative value between min and max (background may be present)
        set.border=5;       % skip border to avoid artefacts of missalignment on edges
        set.cell=floor(set.grid_step/2);    % value to set area to look for maximum before cpcorr
        %set.degree = [];
    case 3 % autoXY_shift with compatible settings to Version 2.1 (04.04.2011, by Volodymyr Cherkas)
        set.fun=varargin{1};
        switch set.fun
            case {'nonreflectivesimilarity','similarity','affine','projective','polynomial','piecewise linear','lwm'}
                set.bin=4;          % default bin for preshift
                set.grid_step=5;    % cpcorr works well within +/-5pixels, thus grid <5 is not recommended
                set.threshold=0.02; % threshold as a relative value between min and max (background may be present)
                set.border=5;       % skip border to avoid artefacts of missalignment on edges
                set.cell=floor(set.grid_step/2);    % value to set area to look for maximum before cpcorr          
            otherwise
                error('Error, transformation function not identified')
        end
    otherwise
        set.fun='nonreflective similarity';
        set.bin=4;          % default bin for preshift (test if works with even)
        set.grid_step=5;    % cpcorr works well within +/-5pixels, thus grid <5 is not recommended
        set.threshold=0.02; % threshold as a relative value between min and max (background may be present)
        set.border=5;       % skip border to avoid artefacts of missalignment on edges
        set.cell=floor(set.grid_step/2);    % value to set area to look for maximum before cpcorr
        set.degree = [];
        for k=3:2:nargin
            switch varargin{k-2}
                case 'fun'
                    set.fun=varargin{k-1};
                case 'bin'
                    set.bin=varargin{k-1};
                case 'grid'
                    set.grid_step=varargin{k-1};
                case 'threshold'
                    set.threshold=varargin{k-1};
                case 'border'
                    set.border=varargin{k-1};
                case 'degree'
                    set.degree=varargin{k-1};
                otherwise
                    error('Error, please check input arguments!')
            end
            set.cell=floor(set.grid_step/2);    % value to set area to look for maximum before cpcorr
        end
end

%% create binned images
% need to normalize images to have cpcorr more efficient - scale to dynamic
% range for integers, or to 1 for floating points
C0=C0/max(C0(:));
C1=C1/max(C1(:));
T0=imresize(C0, 1/set.bin);
T1=imresize(C1, 1/set.bin);
T=T0;%max(T0, T1);
%% create web for binned image (variable constant)
sz=size(T0); % sz(1)=y; sz(2)=x;
[y,x]=ndgrid(set.border:set.grid_step:(sz(1)-set.border), set.border:set.grid_step:(sz(2)-set.border)); 
w(1,:)=reshape(y,numel(y),1); % y; w - initial web
w(2,:)=reshape(x,numel(x),1); % x
clear x y;
%% optimize for points with max intensity within grid
wo=zeros(size(w,2),3);
for i=1:size(w,2)
    [cy, iy]=max(T((w(1,i)-set.cell):(w(1,i)+set.cell),(w(2,i)-set.cell):(w(2,i)+set.cell)));
    [cx, ix]=max(cy);
    wo(i,2)=w(1,i)-set.cell+iy(ix)+1; % wo - optimized web with intensity
    wo(i,1)=w(2,i)-set.cell+ix+1;
    wo(i,3)=cx;
end % warning: wo(:,1)=x, wo(:,2)=y (it is needed for proper cpcorr use)
%% remove underthreshold points
Threshold=set.threshold*(max(T(:))-min(T(:)))+min(T(:));
wo(wo(:,3)<Threshold,:)=[]; % wo(1,:) = y, wo(2,:) = x, wo(3,:) = Intensity
%% optimize web for binned image with cpcorr
wc=cpcorr(wo(:,1:2), wo(:,1:2), T0, T1); % movingPointsAdjusted = cpcorr(movingPoints,fixedPoints,moving,fixed); warning: wc(:,1)=x, wc(:,2)=y (it is needed for proper cpcorr use)
%% remove pixels with no correlation, try preserving ones with matching correlation (when no correction needed for pixel)
wc(:,3:4)=wo(:,1:2); % movingPoints in wc(:,1:2) and fixedPoints in wc(:,3:4)
wc(:,5)=(wo(:,1)==wc(:,1)).*(wo(:,2)==wc(:,2));
wc(wc(:,5)==1,:)=[];
%% scale corrected grid to full resolution
ws(:,1:4)=wc(:,1:4)*set.bin-floor(set.bin/2)+1; % scaled wc
%% optimize web for full resolution images with cpcorr (fine correction)
wf=cpcorr(ws(:,1:2), ws(:,3:4), C0, C1); % final fine web (movingPointsAdjusted = cpcorr(movingPoints,fixedPoints,moving,fixed);)
wf(:,3:4)=ws(:,3:4); % fixed points trace: wo(:,1:2)->wc(:,3:4)->ws(:,3:4)
wf(:,5)=(wf(:,1)==ws(:,1)).*(wf(:,2)==ws(:,2));
wf(wf(:,5)==1,:)=[];
%% create tform with cp2tform
if isempty(set.degree)
  tform=fitgeotrans(wf(:,1:2), wf(:,3:4), set.fun); % transformation function creation
else
  tform=fitgeotrans(wf(:,1:2), wf(:,3:4), set.fun, set.degree); % transformation function creation
end
% C0c = imtransform(C0, tform, 'bicubic','XData',[1 size(C0,2)],'YData',[1 size(C0,1)]);
% Rfixed = imref2d(size(T0));
% T0c = imwarp(T0, tform, 'linear','OutputView', Rfixed);