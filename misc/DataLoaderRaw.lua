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

  self.iterator = 1
end

function DataLoaderRaw:resetIterator()
  self.iterator = 1
end

--[[
  Returns a batch of data:
  - X (N,3,256,256) containing the images as uint8 ByteTensor
  - info table of length N, containing additional information
  The data is iterated linearly in order
--]]
function DataLoaderRaw:getBatch(opt)
  -- may possibly preprocess the image by resizing, cropping
  local seq_length = utils.getopt(opt, 'seq_length', 1000)
  local batch_size = utils.getopt(opt, 'batch_size', 5)
  -- load an image
  local img = image.load(self.files[self.iterator], 3, 'byte')
  local img_raw = image.scale(img, 256, 256)
  local pixel_size = img_raw:size(3)

  local pixels = torch.FloatTensor(seq_length, batch_size, pixel_size+1)


  local infos = {}
  for i=1,batch_size do
    local ri = self.iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterator = ri_next

    -- load the image
    local img = image.load(self.files[ri], 3, 'byte')
    img_batch_raw[i] = image.scale(img, 256, 256)

    -- and record associated info as well
    local info_struct = {}
    info_struct.id = self.ids[ri]
    info_struct.file_path = self.files[ri]
    table.insert(infos, info_struct)
  end

  local data = {}
  data.images = img_batch_raw
  data.bounds = {it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}
  data.infos = infos
  return data
end
