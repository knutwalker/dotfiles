local opt = vim.opt

-- big cursor in insert mode
-- opt.guicursor = ''
-- highlight line with cursor
opt.cursorline = true

-- line numbers
opt.number = true
opt.relativenumber = true

-- 4 space indentation
opt.tabstop = 4
opt.softtabstop = 4
opt.shiftwidth = 4
opt.expandtab = true
opt.autoindent = true
opt.smartindent = true

-- disable line wrapping
opt.wrap = false

-- undo over backups
opt.swapfile = false
opt.backup = false
opt.undodir = os.getenv("HOME") .. "/.local/share/nvim/undodir"
opt.undofile = true

-- search settings
opt.ignorecase = true
opt.smartcase = true
opt.hlsearch = false
opt.incsearch = true

-- pretty and dark colors
opt.termguicolors = true
opt.background = "dark"

-- show whitespace
opt.list = true
opt.listchars = "space:·,lead:·,trail:·,nbsp:☐,tab:▸ ,extends:❯,precedes:❮"

-- always show the sign column
vim.wo.signcolumn = "yes"

-- show columns at 80 and 120 width
opt.colorcolumn = "80,120"

-- Try to keep 8 lines at the bottom
opt.scrolloff = 8

-- faster cursor update time
opt.updatetime = 500

-- allow backspace over autoindent, line breaks, and beyond start of insert
opt.backspace = "indent,eol,start"

-- always show completion menu and don't select automatically
opt.completeopt = "menu,menuone,noinsert,noselect,preview"

-- yank/paste to/from system clipboard
-- we do this with leader mappings instead
-- opt.clipboard:append('unnamedplus')

-- split windows to the right and down by default
opt.splitright = true
opt.splitbelow = true

-- include - when jumping over keywords e.g. with w or b
opt.iskeyword:append("-")

-- Disable command line
opt.cmdheight = 0

-- Disable auto <tab> mapping from copilot. Need to do before plugin is loaded
vim.g.copilot_no_tap_map = true

-- Undotree config, before plugin is loaded
vim.g.undotree_WindowLayout = 4
vim.g.undotree_ShortIndicators = 1
vim.g.undotree_SplitWidth = 40
vim.g.undotree_SetFocusWhenToggle = 1
