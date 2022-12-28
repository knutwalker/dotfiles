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
	init = function()
		vim.keymap.set("n", "<F5>", function()
			require("dap").continue()
		end, { desc = "Debugger: Start" })
		vim.keymap.set("n", "<F17>", function()
			require("dap").terminate()
		end, { desc = "Debugger: Stop" }) -- Shift+F5
		vim.keymap.set("n", "<F29>", function()
			require("dap").restart_frame()
		end, { desc = "Debugger: Restart" }) -- Control+F5
		vim.keymap.set("n", "<F6>", function()
			require("dap").pause()
		end, { desc = "Debugger: Pause" })
		vim.keymap.set("n", "<F20>", function()
			require("dap").toggle_breakpoint()
		end, { desc = "Debugger: Toggle Breakpoint" }) -- Shift + F8
		vim.keymap.set("n", "<F8>", function()
			require("dap").step_over()
		end, { desc = "Debugger: Step Over" })
		vim.keymap.set("n", "<F7>", function()
			require("dap").step_into()
		end, { desc = "Debugger: Step Into" })
		vim.keymap.set("n", "<F19>", function()
			require("dap").step_out()
		end, { desc = "Debugger: Step Out" }) -- Shift+F7

		vim.keymap.set("n", "<leader>Db", function()
			require("dap").toggle_breakpoint()
		end, { desc = "Toggle [B]reakpoint (S-F8)" })
		vim.keymap.set("n", "<leader>DBB", function()
			require("dap").clear_breakpoints()
		end, { desc = "Clear [B]reakpoints" })
		vim.keymap.set("n", "<leader>DB?", function()
			require("dap").list_breakpoints()
		end, { desc = "List [B]reakpoints" })
		vim.keymap.set("n", "<leader>DBc", function()
			require("dap").set_breakpoint(vim.fn.input("Breakpoint condition: "))
		end, { desc = "Set [B]reakpoint with [c]ondition" })
		vim.keymap.set("n", "<leader>DBl", function()
			require("dap").set_breakpoint(nil, nil, vim.fn.input("Log point message: "))
		end, { desc = "Set [B]reakpoint with [l]og message" })

		-- trying to align with lldb commands
		vim.keymap.set("n", "<leader>Dr", function()
			require("dap").continue()
		end, { desc = "[R]un (F5)" })
		vim.keymap.set("n", "<leader>Dc", function()
			require("dap").continue()
		end, { desc = "[C]ontinue (F5)" })
		vim.keymap.set("n", "<leader>Ds", function()
			require("dap").step_into()
		end, { desc = "[S]tep Into (F7)" })
		vim.keymap.set("n", "<leader>Dn", function()
			require("dap").step_over()
		end, { desc = "[N]ext statement (Step Over) (F8)" })
		vim.keymap.set("n", "<leader>Df", function()
			require("dap").step_out()
		end, { desc = "[F]inish current frame (Step Out) (S-F7)" })

		vim.keymap.set("n", "<leader>DS", function()
			require("dap").step_back()
		end, { desc = "[S]tep Back (if possible)" })
		vim.keymap.set("n", "<leader>Dj", function()
			require("dap").down()
		end, { desc = "Go down in current stackframe" })
		vim.keymap.set("n", "<leader>Dk", function()
			require("dap").up()
		end, { desc = "Go up in current stackframe" })
		vim.keymap.set("n", "<leader>Dg", function()
			require("dap").run_to_cursor()
		end, { desc = "[G]o to current cursor" })
		vim.keymap.set("n", "<leader>DG", function()
			require("dap").goto_(tonumber(vim.fn.input("Line > ")))
		end, { desc = "[G]o to and skip to the fiven line" })

		vim.keymap.set("n", "<leader>Dp", function()
			require("dap").pause()
		end, { desc = "Pause (F6)" })
		vim.keymap.set("n", "<leader>DR", function()
			require("dap").restart_frame()
		end, { desc = "Restart (C-F5)" })
		vim.keymap.set("n", "<leader>Dl", function()
			require("dap").run_last()
		end, { desc = "Re-run [l]ast debug session" })

		vim.keymap.set("n", "<leader>Dq", function()
			require("dap").close()
		end, { desc = "Close Session" })
		vim.keymap.set("n", "<leader>DQ", function()
			require("dap").terminate()
		end, { desc = "Terminate Session (S-F5})" })
		vim.keymap.set("n", "<leader>D`", function()
			require("dap").repl.toggle()
		end, { desc = "Toggle REPL" })

		vim.keymap.set("n", "<leader>Du", function()
			require("dapui").toggle()
		end, { desc = "Toggle Debugger UI" })
		vim.keymap.set("n", "<leader>DK", function()
			require("dap.ui.widgets").hover()
		end, { desc = "Debugger Hover" })

		vim.keymap.set("n", "<leader>DD", function()
			require("dap").toggle_breakpoint()
			require("dapui").toggle()
		end, { desc = "Start debug session: Toggle bp & UI" })
	end,
}
