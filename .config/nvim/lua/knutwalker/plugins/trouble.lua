return {
	"folke/trouble.nvim",
	dependencies = { "kyazdani42/nvim-web-devicons" },
	cmd = { "TroubleToggle", "Trouble" },
	init = function()
		vim.keymap.set(
			"n",
			"<leader>f{",
			"<CMD>TroubleToggle workspace_diagnostics<CR>",
			{ silent = true, noremap = true }
		)
		vim.keymap.set(
			"n",
			"<leader>f[",
			"<CMD>TroubleToggle document_diagnostics<CR>",
			{ silent = true, noremap = true }
		)
	end,
	config = {
		auto_open = false,
		use_diagnostic_signs = true,
	},
}
