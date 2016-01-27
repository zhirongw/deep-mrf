--[[
Same as DataLoader but only requires a folder of images.
Does not have an h5 dependency.
Only used at test time.
]]--

local utils = require 'misc.utils'
require 'lfs'
require 'image'

local DataLoaderRaw = torch.class('DataLoaderRaw')

function DataLoaderRaw:__init(opt)
  --- dataLoader of about 100 texture images
  print('DataLoaderRaw loading images from folder: ', opt.folder_path)

  self.files = {}
  self.ids = {}
  -- read in all the filenames from the folder
  --print('listing all images in directory ' .. opt.folder_path)
  local function isImage(f)
    local supportedExt = {'.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.ppm','.PPM'}
    for _,ext in pairs(supportedExt) do
      local _, end_idx =  f:find(ext)
      if end_idx and end_idx == f:len() then
        return true
      end
    end
    return false
  end
  local n = 1
  for file in paths.files(opt.folder_path, isImage) do
    local fullpath = path.join(opt.folder_path, file)
    table.insert(self.files, fullpath)
    table.insert(self.ids, tostring(n)) -- just order them sequentially
    n=n+1
  end

  self.N = #self.files
  print('DataLoaderRaw found ' .. self.N .. ' images')

  self.highres = {}
  self.lowres = {}
  self.kernel = image.gaussian({size=5, sigma=1, normalize=true})
  if opt.color > 0 then self.nChannels = 3 else self.nChannels = 1 end

  for i=1,self.N do
    local img = image.load(self.files[i], self.nChannels, 'float')
    local blurred = image.convolve(img, self.kernel, 'same')
    local resized = image.scale(blurred, math.ceil(blurred:size(3)/3), math.ceil(blurred:size(2)/3), 'bicubic')
    local lowres = image.scale(resized, img:size(3), img:size(2), 'bicubic')
    self.highres[i] = img
    self.lowres[i] = lowres
  end
  self.iterator = 1
end

function DataLoaderRaw:resetIterator()
  self.iterator = 1
end

function DataLoaderRaw:getChannelSize()
  return self.nChannels
end

--[[
  Returns a batch of data:
  - X (N,3,256,256) containing the images as uint8 ByteTensor
  - info table of length N, containing additional information
  The data is iterated linearly in order
--]]
function DataLoaderRaw:getBatch(opt)
  -- may possibly preprocess the image by resizing, cropping
  local patch_size = utils.getopt(opt, 'patch_size', 7)
  local seq_length = patch_size * patch_size - 1
  local batch_size = utils.getopt(opt, 'batch_size', 5)
  -- load an image
  local highres = self.highres[self.iterator]
  local lowres = self.lowres[self.iterator]
  self.iterator = self.iterator + 1
  if self.iterator > self.N then self.iterator = 1 end

  -- two potential schemes, initialize with a border of one pixel in both directions.
  local low_patches, high_patches
  if opt.border == 0 then
    low_patches = torch.zeros(batch_size, self.nChannels, patch_size+2, patch_size+2)
    high_patches = torch.zeros(batch_size, self.nChannels, patch_size+2, patch_size+2)
  else
    low_patches = torch.rand(batch_size, self.nChannels, patch_size+2, patch_size+2)
    high_patches = torch.rand(batch_size, self.nChannels, patch_size+2, patch_size+2)
  end

  --local infos = {}
  for i=1,batch_size do
    local h = torch.random(1, highres:size(2)-patch_size+1)
    local w = torch.random(1, highres:size(3)-patch_size+1)
    -- put the patch in the center.
    high_patches[{i,{},{2,patch_size+1},{2,patch_size+1}}] = highres[{{}, {h, h+patch_size-1}, {w, w+patch_size-1}}]
    low_patches[{i,{},{2,patch_size+1},{2,patch_size+1}}] = lowres[{{}, {h, h+patch_size-1}, {w, w+patch_size-1}}]
    -- and record associated info as well
    -- local info_struct = {}
    -- info_struct.id = self.ids[ri]
    -- info_struct.file_path = self.files[ri]
    -- table.insert(infos, info_struct)
  end
  -- prepare the targets
  local targets = high_patches[{{},{},{2,patch_size+1},{2,patch_size+1}}]:clone()
  targets = targets:view(batch_size, self.nChannels, -1)
  targets = targets:permute(3, 1, 2):contiguous()
  targets = torch.repeatTensor(targets, 2, 1, 1)
  -- prepare the inputs. -n1, left, n2, up, n3, right, n4 down.
  local n1, n2, n3, n4, high_inputs, low_inputs
  n1 = high_patches[{{},{},{2,patch_size+1},{1,patch_size}}]:clone()
  n1 = n1:view(batch_size, self.nChannels, -1)
  n1 = n1:permute(3, 1, 2)
  n2 = high_patches[{{},{},{1,patch_size},{2,patch_size+1}}]:clone()
  n2 = n2:view(batch_size, self.nChannels, -1)
  n2 = n2:permute(3, 1, 2)
  n3 = high_patches[{{},{},{2,patch_size+1},{3,patch_size+2}}]:clone()
  n3 = n3:view(batch_size, self.nChannels, -1)
  n3 = n3:permute(3, 1, 2)
  n4 = high_patches[{{},{},{3,patch_size+2},{2,patch_size+1}}]:clone()
  n4 = n4:view(batch_size, self.nChannels, -1)
  n4 = n4:permute(3, 1, 2)
  high_inputs = torch.cat({n1, n2, n3, n4}, 3)

  n1 = low_patches[{{},{},{2,patch_size+1},{1,patch_size}}]:clone()
  n1 = n1:view(batch_size, self.nChannels, -1)
  n1 = n1:permute(3, 1, 2)
  n2 = low_patches[{{},{},{1,patch_size},{2,patch_size+1}}]:clone()
  n2 = n2:view(batch_size, self.nChannels, -1)
  n2 = n2:permute(3, 1, 2)
  n3 = low_patches[{{},{},{2,patch_size+1},{3,patch_size+2}}]:clone()
  n3 = n3:view(batch_size, self.nChannels, -1)
  n3 = n3:permute(3, 1, 2)
  n4 = low_patches[{{},{},{3,patch_size+2},{2,patch_size+1}}]:clone()
  n4 = n4:view(batch_size, self.nChannels, -1)
  n4 = n4:permute(3, 1, 2)
  low_inputs = torch.cat({n1, n2, n3, n4}, 3)
  local inputs = torch.cat(high_inputs, low_inputs)

  local data = {}
  data.pixels = inputs
  data.targets = targets
  if opt.gpu >= 0 then
    data.pixels = data.pixels:cuda()
    data.targets = data.targets:cuda()
  end
  -- data.infos = infos
  return data
end
