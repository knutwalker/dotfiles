-- This file can be loaded by calling `lua require('plugins')` from your init.vim

-- Only required if you have packer configured as `opt`
vim.cmd([[packadd packer.nvim]])

return require("packer").startup(function(use)
	-- Packer can manage itself
	use("wbthomason/packer.nvim")

	-- Rose-pine colorscheme
	use("rose-pine/neovim")

	-- Github colorscheme
	use("projekt0n/github-nvim-theme")

	-- Tokyo Night colorscheme
	use("folke/tokyonight.nvim")

	-- Tree sitter
	use("nvim-treesitter/nvim-treesitter", { run = ":TSUpdate" })
	use("nvim-treesitter/playground")
	-- Additional text objects via treesitter
	use({
		"nvim-treesitter/nvim-treesitter-textobjects",
		after = "nvim-treesitter",
	})

	-- LSP
	use({
		"VonHeikemen/lsp-zero.nvim",
		requires = {
			-- LSP Support
			{ "neovim/nvim-lspconfig" },
			{ "williamboman/mason.nvim" },
			{ "williamboman/mason-lspconfig.nvim" },

			-- Autocompletion
			{ "hrsh7th/nvim-cmp" },
			{ "hrsh7th/cmp-buffer" },
			{ "hrsh7th/cmp-path" },
			{ "hrsh7th/cmp-nvim-lsp" },
			{ "hrsh7th/cmp-nvim-lsp-signature-help" },
			{ "hrsh7th/cmp-nvim-lua" },
			{ "petertriho/cmp-git" },
			{ "hrsh7th/cmp-emoji" },
			{ "saadparwaiz1/cmp_luasnip" },

			-- Snippets
			{ "L3MON4D3/LuaSnip" },
			{ "rafamadriz/friendly-snippets" },

			-- Progress Report
			{ "j-hui/fidget.nvim" },

			-- -- LSP Signature
			-- { "ray-x/lsp_signature.nvim" },
			--
			-- -- LSP Status
			-- { "nvim-lua/lsp-status.nvim" },
			--
			-- -- LSP Saga
			-- { "glepnir/lspsaga.nvim" },
			--
			-- -- LSP Trouble
			-- { "folke/lsp-trouble.nvim" },

			-- LSP kind symbols for completion items
			{ "onsails/lspkind-nvim" },
		},
	})

	-- Rust LSP
	use("simrat39/rust-tools.nvim")

	-- Better LSP ui
	-- use({ 'glepnir/lspsaga.nvim', branch = 'main' })

	-- Formattesr and Linters
	use("jose-elias-alvarez/null-ls.nvim")
	use("jayp0521/mason-null-ls.nvim")

	-- Debugger
	use("mfussenegger/nvim-dap")
	use("rcarriga/nvim-dap-ui")
	use("jayp0521/mason-nvim-dap.nvim")

	-- Telescope
	use({
		"nvim-telescope/telescope.nvim",
		branch = "0.1.x",
		requires = { "nvim-lua/plenary.nvim" },
	})
	use({
		"nvim-telescope/telescope-fzf-native.nvim",
		run = "gmake",
		cond = vim.fn.executable("gmake") == 1,
	})
	use("nvim-telescope/telescope-ui-select.nvim")

	-- Harpoon
	use("ThePrimeagen/harpoon")

	-- Undotree
	use("mbbill/undotree")

	-- fuGITive
	use("tpope/vim-fugitive")

	-- Zen mode
	use("folke/zen-mode.nvim")

	-- Github Copilot
	use("github/copilot.vim")

	-- Lua line (Status line)
	use({
		"nvim-lualine/lualine.nvim",
		requires = { "kyazdani42/nvim-web-devicons", opt = true },
	})

	-- Add indentation guides even on blank lines
	use("lukas-reineke/indent-blankline.nvim")

	-- 'gc' to comment visual regions/lines
	use("numToStr/Comment.nvim")

	-- Detect tabstop and shiftwidth automatically
	-- use('tpope/vim-sleuth')

	-- Show possible keybindings
	use("folke/which-key.nvim")

	-- Tmux & Split window navigation
	-- use('christoomey/vim-tmux-navigator')

	-- Operate on surroundings
	use("tpope/vim-surround")

	-- Work with various jump and quick lists
	use("tpope/vim-unimpaired")

	-- Sessionize vim
	use("tpope/vim-obsession")

	-- Vim Be Good
	use("ThePrimeagen/vim-be-good")

	-- Read PDFs
	use("makerj/vim-pdf")

	-- Maximize window
	use("szw/vim-maximizer")

	-- startup time measurement
	use("dstein64/vim-startuptime")
end)
