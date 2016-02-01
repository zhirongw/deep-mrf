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
  print('DataLoaderRaw loading images from database: ', opt.folder_path)
  data = matio.load(opt.folder_path, 'data')
  self.N = data['num'][{1,1}]
  print('DataLoaderRaw found ' .. self.N .. ' images')
  local ycb = data['ycb'][{1,1}]
  if ycb == 1 then self.nChannels = 1 else elf.nChannels = 3 end

  self.highres = {}
  self.lowres = {}

  bicubic_loss = 0
  for i=1,self.N do
    local highres = data['highres'][i]:type('torch.FloatTensor'):div(255)
    local lowres = data['lowres'][i]:type('torch.FloatTensor')
    if self.nChannels == 1 then -- only work Y component of YCrCb
      highres = highres[{{1},{},{}}]:clone()
      lowres = lowres[{{1},{},{}}]:clone()
    end
    self.highres[i] = highres:add(opt.shift)
    self.lowres[i] = lowres:add(opt.shift)
    local diff = torch.csub(highres,lowres)
    diff = diff:cmul(diff)
    bicubic_loss = bicubic_loss + torch.mean(diff)
  end
  bicubic_loss = bicubic_loss / 2 / self.N
  print('bicubic baseline is : ', bicubic_loss)
  data = nil -- free it
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
  local patch_size = utils.getopt(opt, 'patch_size', 15)
  local border_size = utils.getopt(opt, 'border_size', 5)
  local seq_length = patch_size * patch_size - 1
  local batch_size = utils.getopt(opt, 'batch_size', 5)
  -- load an image
  local highres = self.highres[self.iterator]
  local lowres = self.lowres[self.iterator]
  self.iterator = self.iterator + 1
  if self.iterator > self.N then self.iterator = 1 end

  -- two potential schemes, initialize with a border of one pixel in both directions.
  local low_patches = torch.zeros(batch_size, self.nChannels, patch_size, patch_size)
  local high_patches
  high_patches = torch.Tensor(batch_size, self.nChannels, patch_size+2, patch_size+2):fill(opt.border)

  --local infos = {}
  for i=1,batch_size do
    local h = torch.random(1, highres:size(2)-patch_size+1)
    local w = torch.random(1, highres:size(3)-patch_size+1)
    -- put the patch in the center.
    high_patches[{i,{},{2,patch_size+1},{2,patch_size+1}}] = highres[{{}, {h, h+patch_size-1}, {w, w+patch_size-1}}]
    low_patches[i] = lowres[{{}, {h, h+patch_size-1}, {w, w+patch_size-1}}]
    -- and record associated info as well
    -- local info_struct = {}
    -- info_struct.id = self.ids[ri]
    -- info_struct.file_path = self.files[ri]
    -- table.insert(infos, info_struct)
  end
  -- prepare the targets
  local targets = high_patches[{{},{},{2,patch_size+1},{2,patch_size+1}}]
  targets = targets[{{},{},{border_size+1,patch_size-border_size},{border_size+1, patch_size-border_size}}]:clone()
  targets = targets:view(batch_size, self.nChannels, -1)
  targets = targets:permute(3, 1, 2):contiguous()
  if opt.num_neighbors == 4 then
    targets = torch.repeatTensor(targets, 2, 1, 1)
  end
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
  if opt.num_neighbors == 3 then
    high_inputs = torch.cat({n1, n2, n3}, 3)
  elseif opt.num_neighbors == 4 then
    n4 = high_patches[{{},{},{3,patch_size+2},{2,patch_size+1}}]:clone()
    n4 = n4:view(batch_size, self.nChannels, -1)
    n4 = n4:permute(3, 1, 2)
    high_inputs = torch.cat({n1, n2, n3, n4}, 3)
  end

  low_patches = low_patches:view(batch_size, self.nChannels, -1)
  local low_inputs = low_patches:permute(3, 1, 2)

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
