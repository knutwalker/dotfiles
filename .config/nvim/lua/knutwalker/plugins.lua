return {
	{
		"tpope/vim-fugitive",
		cmd = "Git",
		init = function()
			vim.keymap.set("n", "<leader>gs", vim.cmd.Git, { desc = "[G]it [S]tatus" })
		end,
	},

	{ "tpope/vim-unimpaired", event = "BufReadPre" },

	{ "tpope/vim-obsession", event = "VeryLazy" },

	{ "makerj/vim-pdf", ft = "pdf" },

	{ "editorconfig/editorconfig-vim", event = "BufReadPre" },

	{ "mg979/vim-visual-multi", branch = "master", event = "BufReadPre" },

	{ "numToStr/Comment.nvim", config = true },

	{ "dstein64/vim-startuptime", cmd = "StartupTime" },

	{ "ThePrimeagen/vim-be-good", cmd = "VimBeGood" },

	{
		"mbbill/undotree",
		cmd = "UndotreeToggle",
		init = function()
			vim.keymap.set("n", "<leader>u", "<cmd>UndotreeToggle<CR>", { desc = "Toggle [U]ndo tree" })
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

	{
		"simrat39/symbols-outline.nvim",
		cmd = { "SymbolsOutline", "SymbolsOutlineOpen" },
		init = function()
			vim.keymap.set({ "n", "v", "i" }, "<F12>", "<cmd>SymbolsOutline<CR>", { desc = "Symbols outline" })
		end,
		config = true,
	},

	{
		"karb94/neoscroll.nvim",
		enabled = false,
		event = "VeryLazy",
		config = {
			-- All these keys will be mapped to their corresponding default scrolling animation
			mappings = { "<C-u>", "<C-d>", "<C-b>", "<C-f>", "<C-y>", "<C-e>", "zt", "zz", "zb" },
			hide_cursor = true, -- Hide cursor while scrolling
			stop_eof = true, -- Stop at <EOF> when scrolling downwards
			respect_scrolloff = false, -- Stop scrolling when the cursor reaches the scrolloff margin of the file
			cursor_scrolls_alone = true, -- The cursor will keep on scrolling even if the window cannot scroll further
			easing_function = nil, -- Default easing function
			pre_hook = nil, -- Function to run before the scrolling animation starts
			post_hook = nil, -- Function to run after the scrolling animation ends
			performance_mode = false, -- Disable "Performance Mode" on all buffers.
		},
	},
}
