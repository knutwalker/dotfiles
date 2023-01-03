return {
	"folke/todo-comments.nvim",
	cmd = { "TodoTrouble", "TodoTelescope" },
	event = "BufReadPost",
	config = {},
	keys = {
		{
			"]t",
			function()
				require("todo-comments").jump_next()
			end,
			desc = "Next todo comment",
		},
		{
			"[t",
			function()
				require("todo-comments").jump_prev()
			end,
			desc = "Previous todo comment",
		},
		{
			"<leader>ft",
			"<cmd>TodoTelescope<CR>",
			desc = "Find [T]odos",
		},
		{
			"<leader>fT",
			"<cmd>TodoTrouble<CR>",
			desc = "Find [T]odos in Touble",
		},
	},
}
