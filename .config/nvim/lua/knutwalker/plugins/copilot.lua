return {
	"github/copilot.vim",
	cmd = "Copilot",
	-- load copilot on InsertEnter
	event = "InsertEnter",
	init = function()
		-- Disable auto <tab> mapping from copilot.
		vim.g.copilot_no_tap_map = true

		-- Don't copilot in Telescope prompts, do in markdown
		vim.g.copilot_filetypes = { ["TelescopePrompt"] = 0, ["markdown"] = 1 }

		-- Use C-j to accept solutions
		vim.keymap.set(
			"i",
			"<C-j>",
			[[copilot#Accept("<CR>")]],
			{ remap = false, silent = true, script = true, expr = true, replace_keycodes = false }
		)

		-- Custom mapping to <leader>c group
		vim.keymap.set("n", "<leader>cl", ":Copilot panel<CR>", { desc = "Open [C]opilot [l]ist" })
		vim.keymap.set("n", "<leader>ce", ":Copilot enable<CR>", { desc = "[C]opilot: [E]nable" })
		vim.keymap.set("n", "<leader>cd", ":Copilot disable<CR>", { desc = "[C]opilot: [D]isable" })
		vim.keymap.set("n", "<leader>cs", ":Copilot status<CR>", { desc = "Show [C]opilot [s]tatus" })
	end,
}
