return {
	"mfussenegger/nvim-dap",
	lazy = true,
	dependencies = {
		{
			"rcarriga/nvim-dap-ui",
			config = function()
				require("dapui").setup({
					icons = { expanded = "▾", collapsed = "▸", current_frame = "☞" },
					layouts = {
						{
							elements = {
								{ id = "scopes", size = 0.45 },
								{ id = "stacks", size = 0.3 },
								{ id = "breakpoints", size = 0.25 },
							},
							size = 0.3,
							position = "left",
						},
						{
							elements = { "repl", "watches" },
							size = 0.3,
							position = "bottom",
						},
					},
					controls = {
						icons = {
							pause = "⏯",
							play = "⏯",
							step_into = "⇣",
							step_over = "⏭",
							step_out = "⇡",
							step_back = "⏮",
							run_last = "↫",
							terminate = "☠",
						},
					},
				})
			end,
		},
		{

			"jayp0521/mason-nvim-dap.nvim",
			config = function()
				local mason_nvim_dap = require("mason-nvim-dap")
				mason_nvim_dap.setup({ automatic_setup = true })
				mason_nvim_dap.setup_handlers({})
			end,
		},
	},
	keys = require("knutwalker.keymap").dap,
}
