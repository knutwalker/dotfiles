return {
	"szw/vim-maximizer",
	keys = { "<C-m>" },
	cmd = "MaximizerToggle",
	config = function()
		vim.keymap.set(
			"n",
			"<C-m>",
			":MaximizerToggle<CR>",
			{ noremap = true, silent = true, desc = "Toggle [m]aximizer" }
		)
	end,
}
