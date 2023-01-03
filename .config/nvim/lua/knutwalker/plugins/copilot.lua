return {
	"zbirenbaum/copilot.lua",
	cmd = "Copilot",
	event = "InsertEnter",
	keys = require("knutwalker.keymap").copilot,
	config = {
		panel = {
			enabled = true,
			auto_refresh = false,
			keymap = {
				jump_prev = "[[",
				jump_next = "]]",
				accept = "<CR>",
				refresh = "gr",
				open = false,
			},
		},
		suggestion = {
			enabled = true,
			auto_trigger = false,
			keymap = {
				accept = "<C-j>",
				accept_word = false,
				accept_line = false,
				next = "<M-]>",
				prev = "<M-[>",
				dismiss = "<C-]>",
			},
		},
		filetypes = { TelescopePrompt = false, markdown = true, yaml = true },
	},
}
