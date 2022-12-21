-- Enable telescope fzf native, if installed
pcall(require("telescope").load_extension, "fzf")
-- Enable ui-select extension, if installed
pcall(require("telescope").load_extension, "ui-select")

local builtin = require("telescope.builtin")

local function bind(mode, key, action, desc)
	vim.keymap.set(mode, key, action, { desc = desc })
end

local function find_git_or_files()
	local opts = {}
	local ok = pcall(builtin.git_files, opts)
	if not ok then
		builtin.find_files(opts)
	end
end

local function find_all_files()
	builtin.find_files({
		hidden = true,
		no_ignore = true,
	})
end

local function grep_all_files()
	builtin.live_grep({
		hidden = true,
		no_ignore = true,
		additional_args = function(args)
			return vim.list_extend(args, { "--hidden", "--no-ignore" })
		end,
	})
end

local function find_in_current_buffer()
	builtin.current_buffer_fuzzy_find(require("telescope.themes").get_dropdown({
		winblend = 10,
		previewer = false,
	}))
end

bind("n", "<leader>/", find_git_or_files, "Search files")
-- bind('n', '<leader>/', builtin.find_files, 'Search all files')
bind("n", "<leader>b", builtin.buffers, "Search [b]uffers")
bind("n", "<leader>e", builtin.oldfiles, "Search [e]dit history")

bind("n", "<leader>ff", builtin.git_files, "[F]ind [f]iles in this git repository")
bind("n", "<leader>fF", builtin.find_files, "[F]ind all [F]iles")
bind("n", "<leader>FF", find_all_files, "[F]ind all [F]iles (including hidden)")
bind("n", "<leader>fg", builtin.live_grep, "[F]ind [g]rep string across files")
bind("n", "<leader>fG", grep_all_files, "[F]ind [g]rep string across files (including hidden)")
bind("n", "<leader>fw", builtin.grep_string, "[F]ind [w]ord under cursor across files")
bind("n", "<leader>f/", find_in_current_buffer, "[F]uzzily [/]search in current buffer]")

bind("n", "<leader>fr", builtin.resume, "[F]ind [r]esume last search")
bind("n", "<leader>fh", builtin.help_tags, "[F]ind [h]elp tags")
bind("n", "<leader>fx", builtin.marks, "[F]ind marks[x]")
bind("n", "<leader>fM", builtin.man_pages, "[F]ind [M]an")
bind("n", "<leader>fr", builtin.registers, "[F]ind [r]egisters")
bind("n", "<leader>fk", builtin.keymaps, "[F]ind [k]eymaps")
bind("n", "<leader>fc", builtin.commands, "[F]ind [c]commands")

bind("n", "<leader>gg", builtin.git_status, "[G]it status")
bind("n", "<leader>gb", builtin.git_branches, "[G]it [b]ranches")
bind("n", "<leader>gc", builtin.git_commits, "[G]it [c]ommits")
bind("n", "<leader>gd", builtin.git_bcommits, "[G]it [d]iff current file")

-- bind('n', '<leader>s', builtin.lsp_document_symbols, 'Search [s]ymbols in this file')
-- bind('n', '<leader>S', builtin.lsp_dynamic_workspace_symbols, 'Search [S]ymbols in this workspace')
-- bind('n', '<leader>fs', builtin.lsp_workspace_symbols, 'Search [S]ymbols in this workspace (once)')
-- bind('n', '<leader>fd', builtin.lsp_definitions, '[F]ind [d]efinitions')
-- bind('n', '<leader>fy', builtin.lsp_type_definitions, '[F]ind t[y]pe definitions')
bind("n", "<leader>{", builtin.diagnostics, "Search Diagnostics")
