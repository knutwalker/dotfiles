-- set leader to space
vim.g.mapleader = " "
vim.g.maplocalleader = " "

local bind = vim.keymap.set

-- Disable default space behavior
bind({ "n", "v" }, "<Space>", "<Nop>", { silent = true })

-- [[ Highlight on yank ]]
-- See `:help vim.highlight.on_yank()`
local highlight_group = vim.api.nvim_create_augroup("YankHighlight", { clear = true })
vim.api.nvim_create_autocmd("TextYankPost", {
	callback = function()
		vim.highlight.on_yank()
	end,
	group = highlight_group,
	pattern = "*",
})

-- Explore opened folder
bind("n", "<leader>pv", vim.cmd.Ex, { desc = "Open [P]roject [V]iew" })

-- move things around in visual mode
bind("v", "J", ":m '>+1<CR>gv=gv", { desc = "Move selection down" })
bind("v", "K", ":m '<-2<CR>gv=gv", { desc = "Move selection up" })

-- Intent lines while keeping visual selection
bind("v", "<", "<gv", { desc = "Indent lines in" })
bind("v", ">", ">gv", { desc = "Indent lines out" })

-- join lines with J while keeping the cursor at the beginning
-- join lines with C-j to move the cursor to the end
bind("n", "J", "mzJ`z", { desc = "Join lines" })

-- jumping around but keep the cursor in the middle
bind("n", "<C-d>", "<C-d>zz", { desc = "Scroll down" })
bind("n", "<C-u>", "<C-u>zz", { desc = "Scroll up" })
bind("n", "n", "nzzzv", { desc = "Next search result" })
bind("n", "N", "Nzzzv", { desc = "Previous search result" })

-- paste with space v without losing the copied content
bind("x", "<leader>p", '"_dp', { desc = "[P]aste without losing copied content" })

-- yank and paste to and from the system clipboard
bind("n", "<leader>y", '"+y', { desc = "[Y]ank to system clipboard" })
bind("v", "<leader>y", '"+y', { desc = "[Y]ank to system clipboard" })
bind("n", "<leader>Y", '"+Y', { desc = "[Y]ank to system clipboard" })

bind("n", "<leader>pp", '"+p', { desc = "[P]aste from system clipboard" })
bind("v", "<leader>pp", '"+p', { desc = "[P]aste from system clipboard" })
bind("n", "<leader>P", '"+P', { desc = "[P]aste from system clipboard" })
bind("v", "<leader>P", '"+P', { desc = "[P]aste from system clipboard" })

bind("n", "<leader>d", '"+d', { desc = "[D]elete to system clipboard" })
bind("v", "<leader>d", '"+d', { desc = "[D]elete to system clipboard" })

-- always trigger ESC when Ctrl-C is pressed
bind("i", "<C-c>", "<Esc>", { desc = "Trigger <ESC> also in visual insert mode" })

-- faster write
bind("n", "<leader>x", ":w<CR>", { desc = "Write buffer" })

-- Never press Q
bind("n", "Q", "<nop>")

-- Open project in tmux session
bind("n", "<C-f>", "<cmd>silent !tmux neww tmux-sessionizer<CR>")

-- Search for selected word
bind(
	"n",
	"<leader>zs",
	[[:%s/\<<C-r><C-w>\>/<C-r><C-w>/gI<Left><Left><Left>]],
	{ desc = "Search/Replace word under cursor" }
)

-- jk to exit insert mode
-- bind('i', 'jk', '<ESC>')

-- space escape to remove highlighting
bind("n", "<leader><ESC>", ":nohl<CR>", { desc = "Remove highlighting" })

-- delete with space x without copying into register
bind("n", "x", '"_x', { desc = "[X] Delete without copying into register" })

-- space + and space - to increment/decrement
bind("n", "<leader>=", "<C-a>", { desc = "Increment number" })
bind("n", "<leader>-", "<C-x>", { desc = "Decrement number" })

-- window management with space w in addition to C-w
bind("n", "<leader>w", "<C-w>", { desc = "Window management" })

-- open new tab
bind("n", "<leader>tc", ":tabnew<CR>", { desc = "[C]reate new tab" })
-- close current tab
bind("n", "<leader>tq", ":tabclose<CR>", { desc = "[Q]uit current tab" })
--  go to next tab
bind("n", "<leader>tn", ":tabn<CR>", { desc = "[N]ext tab" })
--  go to previous tab
bind("n", "<leader>tp", ":tabp<CR>", { desc = "[P]revious tab" })
-- list all tabs
bind("n", "<leader>tl", ":tabs<CR>", { desc = "[L]ist all tabs" })

-- navigate the quickfix list
bind("n", "<leader>cn", ":cnext<CR>zz", { desc = "[N]ext quickfix item" })
bind("n", "<leader>cp", ":cprev<CR>zz", { desc = "[P]revious quickfix item" })
bind("n", "<leader>co", ":copen", { desc = "[O]pen the quickfix list" })
bind("n", "<leader>cq", ":cclose<CR>", { desc = "[Q]uit quickfix list" })
bind("n", "<leader>cc", ":cc", { desc = "Select an item from the quickfix list" })

-- navigate the location list
bind("n", "<leader>ln", ":lnext<CR>zz", { desc = "[N]ext location item" })
bind("n", "<leader>lp", ":lprev<CR>zz", { desc = "[P]revious location item" })
bind("n", "<leader>lo", ":lopen", { desc = "[O]pen the location list" })
bind("n", "<leader>lq", ":lclose<CR>", { desc = "[Q]uit location list" })
bind("n", "<leader>ll", ":ll", { desc = "Select an item from the location list" })
