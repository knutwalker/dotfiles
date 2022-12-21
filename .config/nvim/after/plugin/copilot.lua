-- Use C-j to accept solutions
vim.keymap.set("i", "<C-J>", 'copilot#Accept("\\<CR>")', { remap = false, silent = true, script = true, expr = true })

-- Don't copilot in Telescope prompts, do in markdown
vim.g.copilot_filetypes = { ["TelescopePrompt"] = 0, ["markdown"] = 1 }

-- Custom mapping to <leader>c group
vim.keymap.set("n", "<leader>cl", function()
	vim.cmd(":Copilot panel")
end, { desc = "Open [C]opilot [l]ist" })
vim.keymap.set("n", "<leader>ce", function()
	vim.cmd(":Copilot enable")
end, { desc = "[C]opilot: [E]nable" })
vim.keymap.set("n", "<leader>cd", function()
	vim.cmd(":Copilot disable")
end, { desc = "[C]opilot: [D]isable" })
vim.keymap.set("n", "<leader>cs", function()
	vim.cmd(":Copilot status")
end, { desc = "Show [C]opilot [s]tatus" })
