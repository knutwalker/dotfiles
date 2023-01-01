-- bootstrap from github
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
	vim.fn.system({
		"git",
		"clone",
		"--filter=blob:none",
		"--single-branch",
		"git@github.com:folke/lazy.nvim.git",
		lazypath,
	})
end
vim.opt.runtimepath:prepend(lazypath)

-- load lazy
require("lazy").setup("knutwalker.plugins", {
	defaults = { lazy = false },
	install = { colorscheme = { "tokyonight-moon" } },
	checker = {
		-- do not automatically check for plugin updates
		enabled = false,
	},
	-- performance = {
	-- 	rtp = {
	-- 		disabled_plugins = {
	-- 			"gzip",
	-- 			"matchit",
	-- 			"matchparen",
	-- 			"netrwPlugin",
	-- 			"tarPlugin",
	-- 			"tohtml",
	-- 			"tutor",
	-- 			"zipPlugin",
	-- 		},
	-- 	},
	-- },
})
vim.keymap.set("n", "<leader>lz", "<cmd>:Lazy<cr>")
vim.keymap.set("n", "<leader>ly", "<cmd>:Lazy<cr>")
