return {
	"nvim-treesitter/nvim-treesitter",
	build = ":TSUpdate",
	event = "BufReadPost",
	dependencies = {
		"nvim-treesitter/nvim-treesitter-textobjects",
		{
			"nvim-treesitter/nvim-treesitter-context",
			config = {
				patterns = {
					rust = {
						"impl_item",
						"match",
						"loop",
					},
				},
			},
		},
		{ "nvim-treesitter/playground", cmd = "TSPlaygroundToggle" },
	},
	config = function()
		require("nvim-treesitter.configs").setup({
			-- A list of parser names, or 'all'
			ensure_installed = {
				-- vim related things
				"help",
				"c",
				"lua",
				"vim",
				-- git things
				"diff",
				"git_rebase",
				"gitattributes",
				"gitcommit",
				"gitignore",
				-- proper languages
				"rust",
				"java",
				"kotlin",
				"scala",
				"python",
				"zig",
				-- other languages
				"comment", -- comment tags like TO_DO
				"bash",
				"dockerfile",
				"dot",
				"ebnf",
				"jq",
				"json",
				"make",
				"markdown",
				"markdown_inline",
				"proto",
				"regex",
				"toml",
				-- frontend stuff
				"css",
				"scss",
				"html",
				"javascript",
				"typescript",
			},

			-- Install parsers synchronously (only applied to `ensure_installed`)
			sync_install = false,

			-- Automatically install missing parsers when entering buffer
			-- Recommendation: set to false if you don't have `tree-sitter` CLI installed locally
			auto_install = true,

			highlight = {
				-- `false` will disable the whole extension
				enable = true,

				-- Setting this to true will run `:h syntax` and tree-sitter at the same time.
				-- Set this to `true` if you depend on 'syntax' being enabled (like for indentation).
				-- Using this option may slow down your editor, and you may see some duplicate highlights.
				-- Instead of true it can also be a list of languages
				additional_vim_regex_highlighting = false,
			},

			-- Use tree sitter to power the = indentation operation
			indent = { enable = true },

			-- incrementally increase selection
			incremental_selection = {
				enable = true,
				keymaps = {
					init_selection = "<M-k>",
					node_incremental = "<M-k>",
					node_decremental = "<M-j>",
					scope_incremental = "<M-l>",
				},
			},

			-- Tree sitter powered text objects and navigation
			textobjects = {
				select = {
					enable = true,
					lookahead = true, -- Automatically jump forward to textobj, similar to targets.vim
					keymaps = {
						-- You can use the capture groups defined in textobjects.scm
						["aa"] = { query = "@parameter.outer", desc = "Select outer parameter" },
						["ia"] = { query = "@parameter.inner", desc = "Select inner parameter" },
						["af"] = { query = "@function.outer", desc = "Select outer function" },
						["if"] = { query = "@function.inner", desc = "Select inner function" },
						["ac"] = { query = "@class.outer", desc = "Select outer class" },
						["ic"] = { query = "@class.inner", desc = "Select inner class" },
					},
				},
				move = {
					enable = true,
					set_jumps = true, -- whether to set jumps in the jumplist
					goto_next_start = {
						["]m"] = { query = "@function.outer", desc = "Goto next function" },
						["]]"] = { query = "@class.outer", desc = "Goto next class" },
					},
					goto_next_end = {
						["]M"] = { query = "@function.outer", desc = "Goto the end of the next function" },
						["]["] = { query = "@class.outer", desc = "Goto the end of the next class" },
					},
					goto_previous_start = {
						["[m"] = { query = "@function.outer", desc = "Goto previous function" },
						["[["] = { query = "@class.outer", desc = "Goto previous class" },
					},
					goto_previous_end = {
						["[M"] = { query = "@function.outer", desc = "Goto the end of the previous function" },
						["[]"] = { query = "@class.outer", desc = "Goto the end of the previous class" },
					},
				},
				swap = {
					enable = true,
					swap_next = {
						["<leader>a"] = "@parameter.inner",
					},
					swap_previous = {
						["<leader>A"] = "@parameter.inner",
					},
				},
				lsp_interop = {
					enable = true,
					border = "none",
					floating_preview_opts = {},
					peek_definition_code = {
						["gdf"] = "@function.outer",
						["gdF"] = "@class.outer",
					},
				},
			},
		})
	end,
}
