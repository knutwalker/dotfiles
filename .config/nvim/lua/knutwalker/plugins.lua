return {
	{
		"mbbill/undotree",
		cmd = "UndotreeToggle",
		init = function()
			vim.keymap.set("n", "<leader>u", vim.cmd.UndotreeToggle, { desc = "Toggle [U]ndo tree" })
		end,
	},

	{
		"tpope/vim-fugitive",
		cmd = "Git",
		init = function()
			vim.keymap.set("n", "<leader>gs", vim.cmd.Git, { desc = "[G]it [S]tatus" })
		end,
	},

	{
		"lukas-reineke/indent-blankline.nvim",
		config = function()
			require("indent_blankline").setup({
				ohar = "",
				space_char_blankline = " ",
				show_current_context = true,
				show_current_context_start = true,
				show_trailing_blankline_indent = true,
			})
		end,
	},

	{
		"numToStr/Comment.nvim",
		config = true,
	},

	{
		"dstein64/vim-startuptime",
		cmd = "StartupTime",
	},

	{ "ThePrimeagen/vim-be-good", cmd = "VimBeGood" },

	{
		"folke/which-key.nvim",
		config = function()
			require("which-key").setup()
		end,
	},

	{
		"kylechui/nvim-surround",
		tag = "v1.0.0",
		event = "BufReadPre",
		config = true,
	},

	{
		"windwp/nvim-autopairs",
		event = "InsertEnter",
		config = function()
			require("nvim-autopairs").setup({
				check_ts = true,
			})
		end,
	},

	{ "tpope/vim-unimpaired", event = "BufReadPre" },
	{ "tpope/vim-obsession", event = "VeryLazy" },

	{ "makerj/vim-pdf", ft = "pdf" },
}
