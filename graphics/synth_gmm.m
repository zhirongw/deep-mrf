%  Texture Synthesis using a gaussian mixture model
%  Author: Alex Rubinsteyn (alex.rubinsteyn at gmail)


%  This program is free software; you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation; either version 2 of the License, or
%  (at your option) any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software
%  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
% 02110-1301  USA

%%
function [Image,time, M,V] = synth_gmm(filename, winsize, newRows, newCols)
tic
MaxErrThreshold = 0.1;

rawSample = im2double(imread(filename));

% only works with gray image
if ndims(rawSample) > 2
    sample = rgb2gray(rawSample);
else
    sample = rawSample;
end


[rows, cols] = size(sample);

halfWindow = (winsize - 1) / 2;

npixels = newRows * newCols;
Image = zeros(newRows, newCols);


%% split the image into patch columns and run EM to train a gaussian

all_patches = im2col(sample, [winsize winsize], 'sliding')';

k = 12 * floor(size(all_patches, 1) / 150);

[M, V, Cholesky] = EstimateDensity(all_patches, k);


%% initialize new texture with a random 3x3 patch from the sample
randRow = ceil(rand() * (rows - 2));
randCol = ceil(rand() * (cols - 2));

seedSize = 3;

seedRows = ceil(newRows/2):ceil(newRows/2)+seedSize-1;
seedCols = ceil(newCols/2):ceil(newCols/2)+seedSize-1;

Image(seedRows, seedCols) = sample(randRow:randRow+seedSize-1, randCol:randCol+seedSize-1);


nfilled = seedSize * seedSize;

filled = repmat(false, [newRows newCols]);
filled(seedRows, seedCols) = repmat(true, [3 3]);



%% the main act

gaussMask = fspecial('gaussian',winsize, winsize/6.4);
nskipped = 0;

while nfilled < npixels
    progress = false;

    [pixelRows, pixelCols] = GetUnfilledNeighbors(filled, winsize);

     for i = 1:length(pixelRows)
        pixelRow = pixelRows(i);
        pixelCol = pixelCols(i);

        rowRange = pixelRow-halfWindow:pixelRow+halfWindow;
        colRange =  pixelCol - halfWindow:pixelCol + halfWindow;

        deadRows = rowRange < 1 | rowRange > newRows;
        deadCols = colRange < 1 | colRange > newCols;


        if sum(deadRows) + sum(deadCols) > 0
            safeRows = rowRange(~deadRows);
            safeCols = colRange(~deadCols);

            template = zeros(winsize, winsize);
            template(~deadRows, ~deadCols) = Image(safeRows, safeCols);

            validMask = repmat(false, [winsize winsize]);
            validMask(~deadRows, ~deadCols) = filled(safeRows, safeCols);

        else
            template = Image(rowRange, colRange);
            validMask = filled(rowRange, colRange);

        end

       [bestPatch, bestSSD] = FindMatches(template, validMask, gaussMask, M, V, Cholesky, 100);

         if bestSSD < MaxErrThreshold
             squarePatch = reshape(bestPatch, [winsize winsize]);

             Image(pixelRow, pixelCol) = squarePatch(halfWindow+1, halfWindow+1);
             filled(pixelRow, pixelCol) = true;
             nfilled = nfilled + 1;
             progress = true;
         else
             nskipped = nskipped + 1;
         end
    end


    disp(sprintf('Pixels filled: %d / %d', nfilled, npixels));
    if ~progress

        MaxErrThreshold = MaxErrThreshold * 1.1;
        disp(sprintf('Incrementing error tolerance to %d', MaxErrThreshold));
    end
end
toc
time = toc;
figure;
imshow(Image);

%% Get pixels at edge of synthesized texture
 function [pixelRows, pixelCols] = GetUnfilledNeighbors(filled, winsize)
    border = bwmorph(filled,'dilate')-filled;

    [pixelRows, pixelCols] = find(border);
    len = length(pixelRows);

     %randomly permute candidate pixels
     randIdx = randperm(len);
     pixelRows = pixelRows(randIdx);
     pixelCols = pixelCols(randIdx);

     %sort by number of neighbors
     filledSums = colfilt(filled, [winsize winsize], 'sliding', @sum);
     numFilledNeighbors = filledSums( sub2ind(size(filled), pixelRows, pixelCols) );
     [sorted, sortIndex] = sort(numFilledNeighbors, 1, 'descend');

     pixelRows = pixelRows(sortIndex);
     pixelCols = pixelCols(sortIndex);


%% Sample patches and return one that best matches template
function [bestPatch, bestSSD] = FindMatches (template, validMask, gaussMask, M, V, Cholesky, npatches)

valid_vec = validMask(:);

subM = M(valid_vec, :);
subV = V(valid_vec, valid_vec, :);

% length of probs: number of mixtures
probs = mvnpdf(template(valid_vec)', subM', subV);

probs = probs / sum(probs);

[pixels_per_patch, nclusters] = size(M);
patches = zeros(pixels_per_patch, npatches);

idx = 1;
for i = 1:nclusters
    n = floor(npatches * probs(i));
    if n > 0
        range = idx:(idx+n-1);
        result =  mvnrnd(M(:, i)', Cholesky(:, :, i), n)';
        patches(:, range) = result;
        idx = idx + n;
    end
end

totalWeight = sum(sum(gaussMask(validMask)));

mask = (gaussMask .* validMask) / totalWeight;
mask_vec = mask(:)';

template_vec = reshape(template, [pixels_per_patch 1]);
templates = repmat(template_vec, [1 npatches]);

SSD = mask_vec * (templates - patches).^2;

[bestSSD, bestIdx] = min(SSD);
bestPatch = patches(:, bestIdx);

%%
    function [M, V, Cholesky] = EstimateDensity(all_patches, k)

    if (k > 1)
        warning('off', 'MATLAB:log:logOfZero');
        warning('off', 'MATLAB:illConditionedMatrix');

        disp(['Estimating mixture of gaussians (k=' int2str(k) ')']);

        [W,M,V] = EM_GM(all_patches,k, .1, 500); %,ltol,maxiter,pflag,Init)

        Finite = all(isfinite(W)) || all(all(isfinite(M))) || all(all(all(isfinite(V))));

        AnyGoodDecomposition = false;
        AnyFullRank = false;

        if Finite
            desired_rank =  size(all_patches, 2);
            fullrank_idx = repmat(true, [k 1]);

           nfullrank = 0;

            for i = 1:k
                   if rank(V(:, :, i)) ~= desired_rank
                      fullrank_idx(i) = false;
                      disp(sprintf('Modifying covariance matrix #%d due to insufficient rank', i));
                      V(:, :, i) = V(:, :, i) + 0.1*eye(size(V(:, :, i)));
                   else
                       nfullrank = nfullrank + 1;
                   end
            end
            AnyFullRank = nfullrank > 0;

            if AnyFullRank
%
%                 V = V(:, :, fullrank_idx);
%                 M = M(:,  fullrank_idx);

                Cholesky  = zeros(size(V));
                newK = size(V, 3);

                decomposition_idx = repmat(true, [newK 1]);

                for i = 1:newK
                    [U, success] = decompose(V(:,:, i));
                    Cholesky(:, :, i) = U;

                    if ~success
                        decomposition_idx(i) = false;
                        disp(sprintf('Skipping covariance matrix #%d due to bad decomposition', i));
                    end
                end

                Cholesky = Cholesky(:, :, decomposition_idx);
                V = V(:, :, decomposition_idx);
                M = M(:, decomposition_idx);

                AnyGoodDecomposition = ~isempty(Cholesky);
            end
        end

        warning('on', 'MATLAB:log:logOfZero');
        warning('on', 'MATLAB:illConditionedMatrix');

        if ~Finite || ~AnyFullRank || ~AnyGoodDecomposition
            disp(['Retrying due to numerical craziness - finite: ' int2str(Finite) ', full rank: ' int2str(AnyFullRank) ', decomp: ' int2str(AnyGoodDecomposition)]);
            [M,V,Cholesky] = EstimateDensity(all_patches, k-1);
        end
    else
        error('Ran out of gaussians, aborting');
    end



%%
% try cholesky decomposition, if it fails, diagonalize fails...well, we're toast.
function [U, success] = decompose(V)
    warning('off', 'MATLAB:log:logOfZero');
    warning('off', 'MATLAB:illConditionedMatrix');
    success = true;
     U = zeros(size(V));
    if ~all(all(isfinite(V)))
        success = false;
    else
        try
          U=chol(V);
        catch
            [E,Lambda]=eig(V);
            if (min(diag(Lambda))<0)
                %error('Covariance must be positive semi-definite.')
                success = false;

            else
                U = sqrt(Lambda)*E';
            end
        end
    end




%%
% Copyright (C) 2003 Iain Murray
% -------------------------------------------------------------------------
% Modified by Alex Rubinsteyn to accept decomposed sigma and stripped out
% some code I didn't need

function s = mvnrnd(mu,U,K)
if nargin==3
	mu=repmat(mu,K,1);
end
[n,d]=size(mu);
s = randn(n,d)*U + mu;
