local dap = require("dap")
local dapui = require("dapui")
local bind = vim.keymap.set

bind("n", "<F5>", dap.continue, { desc = "Debugger: Start" })
bind("n", "<F17>", dap.terminate, { desc = "Debugger: Stop" }) -- Shift+F5
bind("n", "<F29>", dap.restart_frame, { desc = "Debugger: Restart" }) -- Control+F5
bind("n", "<F6>", dap.pause, { desc = "Debugger: Pause" })
bind("n", "<F20>", dap.toggle_breakpoint, { desc = "Debugger: Toggle Breakpoint" }) -- Shift + F8
bind("n", "<F8>", dap.step_over, { desc = "Debugger: Step Over" })
bind("n", "<F7>", dap.step_into, { desc = "Debugger: Step Into" })
bind("n", "<F19>", dap.step_out, { desc = "Debugger: Step Out" }) -- Shift+F7

bind("n", "<leader>Db", dap.toggle_breakpoint, { desc = "Toggle [B]reakpoint (S-F8)" })
bind("n", "<leader>DBB", dap.clear_breakpoints, { desc = "Clear [B]reakpoints" })
bind("n", "<leader>DB?", dap.list_breakpoints, { desc = "List [B]reakpoints" })
bind("n", "<leader>DBc", function()
	dap.set_breakpoint(vim.fn.input("Breakpoint condition: "))
end, { desc = "Set [B]reakpoint with [c]ondition" })
bind("n", "<leader>DBl", function()
	dap.set_breakpoint(nil, nil, vim.fn.input("Log point message: "))
end, { desc = "Set [B]reakpoint with [l]og message" })

-- trying to align with lldb commands
bind("n", "<leader>Dr", dap.continue, { desc = "[R]un (F5)" })
bind("n", "<leader>Dc", dap.continue, { desc = "[C]ontinue (F5)" })
bind("n", "<leader>Ds", dap.step_into, { desc = "[S]tep Into (F7)" })
bind("n", "<leader>Dn", dap.step_over, { desc = "[N]ext statement (Step Over) (F8)" })
bind("n", "<leader>Df", dap.step_out, { desc = "[F]inish current frame (Step Out) (S-F7)" })

bind("n", "<leader>DS", dap.step_back, { desc = "[S]tep Back (if possible)" })
bind("n", "<leader>Dj", dap.down, { desc = "Go down in current stackframe" })
bind("n", "<leader>Dk", dap.up, { desc = "Go up in current stackframe" })
bind("n", "<leader>Dg", dap.run_to_cursor, { desc = "[G]o to current cursor" })
bind("n", "<leader>DG", function()
	dap.goto_(tonumber(vim.fn.input("Line > ")))
end, { desc = "[G]o to and skip to the fiven line" })

bind("n", "<leader>Dp", dap.pause, { desc = "Pause (F6)" })
bind("n", "<leader>DR", dap.restart_frame, { desc = "Restart (C-F5)" })
bind("n", "<leader>Dl", dap.run_last, { desc = "Re-run [l]ast debug session" })

bind("n", "<leader>Dq", dap.close, { desc = "Close Session" })
bind("n", "<leader>DQ", dap.terminate, { desc = "Terminate Session (S-F5})" })
bind("n", "<leader>D`", dap.repl.toggle, { desc = "Toggle REPL" })

bind("n", "<leader>Du", dapui.toggle, { desc = "Toggle Debugger UI" })
bind("n", "<leader>DK", require("dap.ui.widgets").hover, { desc = "Debugger Hover" })

bind("n", "<leader>DD", function()
	dap.toggle_breakpoint()
	dapui.toggle()
end, { desc = "Start debug session: Toggle bp & UI" })

local mason_nvim_dap = require("mason-nvim-dap")
mason_nvim_dap.setup({ automatic_setup = true })
mason_nvim_dap.setup_handlers({})

dapui.setup({
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
