--[[
Same as DataLoader but only requires a folder of images.
Does not have an h5 dependency.
Only used at test time.
]]--

local utils = require 'misc.utils'
local matio = require 'matio'
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

  self.images = {}
  if opt.color > 0 then self.nChannels = 3 else self.nChannels = 1 end

  for i=1,self.N do
    local img = image.load(self.files[i], self.nChannels, 'float')
    if img:dim() == 2 then img = img:resize(1, img:size(1), img:size(2)) end
    if img:size(2) > opt.img_size or img:size(3) > opt.img_size then
      local factor = math.max(opt.img_size/img:size(2), opt.img_size/img:size(3))
      img = image.scale(img, math.ceil(img:size(2)*factor-0.5), math.ceil(img:size(3)*factor-0.5))
    end
    self.images[i] = img:add(opt.shift)
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
  local crop_size = utils.getopt(opt, 'crop_size', 64)
  local batch_size = utils.getopt(opt, 'batch_size', 4)

  local images = torch.Tensor(batch_size, self.nChannels, crop_size, crop_size)
  --local infos = {}

  for i=1,batch_size do
    local im = self.images[self.iterator]
    self.iterator = self.iterator + 1
    if self.iterator > self.N then self.iterator = 1 end

    local h = torch.random(1, im:size(2)-crop_size+1)
    local w = torch.random(1, im:size(3)-crop_size+1)
    -- put the patch in the center.
    images[i] = im[{{}, {h, h+crop_size-1}, {w, w+crop_size-1}}]
    -- and record associated info as well
    -- local info_struct = {}
    -- info_struct.id = self.ids[ri]
    -- info_struct.file_path = self.files[ri]
    -- table.insert(infos, info_struct)
  end

  if opt.gpu >= 0 then
    images = images:cuda()
  end

  return images
end
