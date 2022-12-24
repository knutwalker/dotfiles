return {
	"folke/zen-mode.nvim",
	cmd = "ZenMode",
	init = function()
		vim.keymap.set("n", "<leader>zz", function()
			require("zen-mode").toggle()
			vim.wo.wrap = false
			ColorMyPencils()
		end, { desc = "Toggle [Z]en mode" })
	end,

	config = function()
		require("zen-mode").setup({
			window = {
				width = 90,
				options = {
					number = true,
					relativenumber = true,
				},
			},
			plugins = {
				gitsigns = true,
				tmux = true,
				kitty = { enabled = false, font = "+2" },
			},
		})
	end,
}
