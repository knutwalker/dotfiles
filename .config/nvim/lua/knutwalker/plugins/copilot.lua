return {
	"zbirenbaum/copilot.lua",
	cmd = "Copilot",
	event = "InsertEnter",
	init = function()
		-- Custom mapping to <leader>C group
		vim.keymap.set("n", "<leader>Cs", ":Copilot status<CR>", { desc = "Show [C]opilot [s]tatus" })
		vim.keymap.set("n", "<leader>Cv", ":Copilot version<CR>", { desc = "Show [C]opilot [v]ersion" })
		vim.keymap.set("n", "<leader>Cl", ":Copilot panel<CR>", { desc = "Open [C]opilot [l]ist" })
		vim.keymap.set("n", "<leader>Ca", function()
			require("copilot.suggestion").toggle_auto_trigger()
		end, { desc = "Toggle [C]opilot [a]uto-trigger" })
		vim.keymap.set("n", "<leader>CC", ":Copilot toggle<CR>", { desc = "Toggle [C]opilot" })
		vim.keymap.set("n", "<leader>Cd", ":CopilotDetach<CR>", { desc = "[D]etach [C]opilot from buffer" })
		vim.keymap.set("n", "<leader>CS", ":CopilotStop<CR>", { desc = "[S]top [C]opilot" })
	end,
	config = function()
		require("copilot").setup({
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
		})
	end,
}
