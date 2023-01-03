return {
	"folke/zen-mode.nvim",
	cmd = "ZenMode",
	keys = require("knutwalker.keymap").zen_mode,
	config = {
		window = {
			width = 90,
			options = {
				number = true,
				relativenumber = true,
			},
		},
		plugins = {
			gitsigns = true,
			tmux = true,
			kitty = { enabled = false, font = "+2" },
		},
	},
}
