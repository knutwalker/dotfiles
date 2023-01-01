return {
	"nvim-telescope/telescope.nvim",
	branch = "0.1.x",
	cmd = { "Telescope", "Tel" },
	dependencies = {
		"nvim-lua/plenary.nvim",
		"nvim-telescope/telescope-ui-select.nvim",
		"nvim-telescope/telescope-symbols.nvim",
		{ "nvim-telescope/telescope-fzf-native.nvim", build = "gmake" },
	},
	init = function()
		local function bind(mode, key, action, desc)
			vim.keymap.set(mode, key, action, { desc = desc })
		end

		bind("n", "<leader>/", function()
			local builtin = require("telescope.builtin")
			local opts = {}
			local ok = pcall(builtin.git_files, opts)
			if not ok then
				builtin.find_files(opts)
			end
		end, "Search files")

		bind("n", "<leader>b", function()
			require("telescope.builtin").buffers()
		end, "Search [b]uffers")

		bind("n", "<leader>e", function()
			require("telescope.builtin").oldfiles()
		end, "Search [e]dit history")

		bind("n", "<leader>ff", function()
			require("telescope.builtin").find_files()
		end, "[F]ind all [F]iles")

		bind("n", "<leader>fF", function()
			require("telescope.builtin").find_files({
				hidden = true,
				no_ignore = true,
			})
		end, "[F]ind all [F]iles (including hidden)")

		bind("n", "<leader>fg", function()
			require("telescope.builtin").live_grep()
		end, "[F]ind [g]rep string across files")

		bind("n", "<leader>fG", function()
			require("telescope.builtin").live_grep({
				hidden = true,
				no_ignore = true,
				additional_args = function(args)
					return vim.list_extend(args, { "--hidden", "--no-ignore" })
				end,
			})
		end, "[F]ind [g]rep string across files (including hidden)")

		bind("n", "<leader>fw", function()
			require("telescope.builtin").grep_string()
		end, "[F]ind [w]ord under cursor across files")

		bind("n", "<leader>f/", function()
			require("telescope.builtin").current_buffer_fuzzy_find(require("telescope.themes").get_dropdown({
				winblend = 10,
				previewer = false,
			}))
		end, "[F]uzzily [/]search in current buffer]")

		bind("n", "<leader>fr", function()
			require("telescope.builtin").resume()
		end, "[F]ind [r]esume last search")

		bind("n", "<leader>fh", function()
			require("telescope.builtin").help_tags()
		end, "[F]ind [h]elp tags")

		bind("n", "<leader>fx", function()
			require("telescope.builtin").marks()
		end, "[F]ind marks[x]")

		bind("n", "<leader>fM", function()
			require("telescope.builtin").man_pages()
		end, "[F]ind [M]an")

		bind("n", "<leader>fr", function()
			require("telescope.builtin").registers()
		end, "[F]ind [r]egisters")

		bind("n", "<leader>fk", function()
			require("telescope.builtin").keymaps()
		end, "[F]ind [k]eymaps")

		bind("n", "<leader>fc", function()
			require("telescope.builtin").commands()
		end, "[F]ind [c]commands")

		bind("n", "<leader>fe", ":Telescope symbols<CR>", "[F]ind [e]mojis")

		bind("n", "<leader>gg", function()
			require("telescope.builtin").git_status()
		end, "[G]it status")

		bind("n", "<leader>gb", function()
			require("telescope.builtin").git_branches()
		end, "[G]it [b]ranches")

		bind("n", "<leader>gc", function()
			require("telescope.builtin").git_commits()
		end, "[G]it [c]ommits")

		bind("n", "<leader>gd", function()
			require("telescope.builtin").git_bcommits()
		end, "[G]it [d]iff current file")

		-- bind('n', '<leader>s', function() require("telescope.builtin").lsp_document_symbols() end, 'Search [s]ymbols in this file')
		-- bind('n', '<leader>S', function() require("telescope.builtin").lsp_dynamic_workspace_symbols() end, 'Search [S]ymbols in this workspace')
		-- bind('n', '<leader>fs', function() require("telescope.builtin").lsp_workspace_symbols() end, 'Search [S]ymbols in this workspace (once)')
		-- bind('n', '<leader>fd', function() require("telescope.builtin").lsp_definitions() end, '[F]ind [d]efinitions')
		-- bind('n', '<leader>fy', function() require("telescope.builtin").lsp_type_definitions() end, '[F]ind t[y]pe definitions')
		bind("n", "<leader>{", function()
			require("telescope.builtin").diagnostics()
		end, "Search Diagnostics")
	end,
	config = function()
		local telescope = require("telescope")
		telescope.setup({
			defaults = {
				layout_strategy = "horizontal",
				layout_config = {
					prompt_position = "top",
					width = 0.9,
					preview_cutoff = 0,
				},
				sorting_strategy = "ascending",
				mappings = {
					i = {
						["<c-t>"] = function(...)
							require("trouble.providers.telescope").open_with_trouble(...)
						end,
					},
					n = {
						["<c-t>"] = function(...)
							require("trouble.providers.telescope").open_with_trouble(...)
						end,
					},
				},
			},
			pickers = {
				buffers = {
					sort_mru = true,
				},
			},
		})

		telescope.load_extension("fzf")
		telescope.load_extension("ui-select")
	end,
}
