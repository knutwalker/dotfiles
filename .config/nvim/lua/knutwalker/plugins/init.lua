local keymap = require("knutwalker.keymap")

return {
	{ "tpope/vim-fugitive", cmd = "Git", keys = keymap.git },

	{ "tpope/vim-unimpaired", event = "BufReadPre" },

	{ "tpope/vim-obsession", event = "VeryLazy" },

	{ "makerj/vim-pdf", ft = "pdf" },

	{ "editorconfig/editorconfig-vim", event = "BufReadPre" },

	{ "mg979/vim-visual-multi", branch = "master", event = "BufReadPre" },

	{ "numToStr/Comment.nvim", config = true },

	{ "dstein64/vim-startuptime", cmd = "StartupTime" },

	{ "ThePrimeagen/vim-be-good", cmd = "VimBeGood" },

	{ "ThePrimeagen/harpoon", keys = keymap.harpoon },

	{ "mbbill/undotree", cmd = "UndotreeToggle", keys = keymap.undotree },

	{ "folke/which-key.nvim", config = true },

	{ "kylechui/nvim-surround", tag = "v1.0.0", event = "BufReadPre", config = true },

	{ "windwp/nvim-autopairs", event = "InsertEnter", config = { check_ts = true } },

	{ "szw/vim-maximizer", keys = keymap.maximizer, cmd = "MaximizerToggle" },

	{
		"lukas-reineke/indent-blankline.nvim",
		config = {
			ohar = "",
			space_char_blankline = " ",
			show_current_context = true,
			show_current_context_start = true,
			show_trailing_blankline_indent = true,
		},
	},

	{
		"simrat39/symbols-outline.nvim",
		cmd = { "SymbolsOutline", "SymbolsOutlineOpen" },
		keys = keymap.symbols_outline,
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

	{
		"saecki/crates.nvim",
		event = { "BufRead Cargo.toml" },
		requires = { { "nvim-lua/plenary.nvim" } },
		keys = keymap.crates,
		config = true,
	},

	{
		"ThePrimeagen/refactoring.nvim",
		keys = keymap.refactoring,
		config = function()
			require("refactoring").setup({})
			require("telescope").load_extension("refactoring")
		end,
	},

	{
		"folke/todo-comments.nvim",
		cmd = { "TodoTrouble", "TodoTelescope" },
		event = "BufReadPost",
		config = true,
		keys = keymap.todo_comments,
	},

	{
		"folke/trouble.nvim",
		dependencies = { "kyazdani42/nvim-web-devicons" },
		cmd = { "TroubleToggle", "Trouble" },
		keys = keymap.trouble,
		config = {
			auto_open = false,
			use_diagnostic_signs = true,
		},
	},
}
