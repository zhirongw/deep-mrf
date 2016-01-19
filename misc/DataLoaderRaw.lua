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
  print('listing all images in directory ' .. opt.folder_path)
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

  -- how about working on the first texture? D1.png
  self.iterator = 1
  self.images = {}
  print('training on image: '..self.files[self.iterator])
  if opt.color > 0 then self.nChannels = 3 else self.nChannels = 1 end

  local img = image.load(self.files[self.iterator], self.nChannels, 'float')
  img = image.scale(img, opt.img_size, opt.img_size)
  img = img:resize(self.nChannels, opt.img_size, opt.img_size)
  -- print(img[{{},1,1}])
  self.images[self.iterator] = img
  self.nHeight = img:size(2)
  self.nWidth = img:size(3)
end

function DataLoaderRaw:resetIterator()
  --self.iterator = 1
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
  local img = self.images[self.iterator]

  -- two potential schems
  local patches
  if opt.border == 0 then
    patches = torch.zeros(batch_size, self.nChannels, patch_size+1, patch_size+1)
  else
    patches = torch.rand(batch_size, self.nChannels, patch_size+1, patch_size+1)
  end

  --local infos = {}
  for i=1,batch_size do
    local h = torch.random(1, self.nHeight-patch_size+1)
    local w = torch.random(1, self.nWidth-patch_size+1)

    patches[{i,{},{2,patch_size+1},{2,patch_size+1}}] = img[{{}, {h, h+patch_size-1}, {w, w+patch_size-1}}]
    -- and record associated info as well
    -- local info_struct = {}
    -- info_struct.id = self.ids[ri]
    -- info_struct.file_path = self.files[ri]
    -- table.insert(infos, info_struct)
  end
  -- prepare the targets
  local targets = patches[{{},{},{2,patch_size+1},{2,patch_size+1}}]:clone()
  targets = targets:view(batch_size, self.nChannels, -1)
  targets = targets:permute(3, 1, 2)
  -- prepare the inputs. -n1, left, n2, up.
  local n1 = patches[{{},{},{2,patch_size+1},{1,patch_size}}]:clone()
  n1 = n1:view(batch_size, self.nChannels, -1)
  n1 = n1:permute(3, 1, 2)
  local n2 = patches[{{},{},{1,patch_size},{2,patch_size+1}}]:clone()
  n2 = n2:view(batch_size, self.nChannels, -1)
  n2 = n2:permute(3, 1, 2)
  local inputs = torch.cat(n1, n2, 3)

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
