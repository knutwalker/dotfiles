-- Unless you are still migrating, remove the deprecated commands from v1.x
vim.cmd([[ let g:neo_tree_remove_legacy_commands = 1 ]])

return {
	"nvim-neo-tree/neo-tree.nvim",
	branch = "v2.x",
	cmd = "Neotree",
	dependencies = {
		"nvim-lua/plenary.nvim",
		"MunifTanjim/nui.nvim",
	},
	keys = {
		{
			"<leader><leader>",
			function()
				require("neo-tree.command").execute({ toggle = true })
			end,
			desc = "Open [F]ile[t]ree",
		},
	},
	config = {
		close_if_last_window = true,
		sort_case_insensitive = true,
		filesystem = {
			follow_current_file = true,
			hijack_netrw_behavior = "open_current",
		},
	},
}
