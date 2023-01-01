return {
	"ThePrimeagen/refactoring.nvim",
	lazy = true,
	config = function()
		require("refactoring").setup({})
		require("telescope").load_extension("refactoring")
	end,
	init = function()
		local bind = vim.keymap.set

		bind({ "n", "v" }, "<leader>rs", function()
			require("refactoring").select_refactor()
		end, { noremap = true, silent = true, expr = false, desc = "Refactor [S]elect" })

		bind(
			{ "n", "v" },
			"<leader>r/",
			"<Esc><cmd>lua require('telescope').extensions.refactoring.refactors()<CR>",
			{ noremap = true, desc = "Search available refactors" }
		)

		bind(
			"v",
			"<leader>rf",
			[[ <Esc><Cmd>lua require('refactoring').refactor('Extract Function')<CR>]],
			{ noremap = true, silent = true, expr = false, desc = "Refactor Extract [f]unction" }
		)
		bind(
			"v",
			"<leader>rF",
			[[ <Esc><Cmd>lua require('refactoring').refactor('Extract Function To File')<CR>]],
			{ noremap = true, silent = true, expr = false, desc = "Refactor function to [F]ile" }
		)
		bind(
			"v",
			"<leader>rv",
			[[ <Esc><Cmd>lua require('refactoring').refactor('Extract Variable')<CR>]],
			{ noremap = true, silent = true, expr = false, desc = "Refactor Extract [v]ariable" }
		)
		bind(
			"v",
			"<leader>ri",
			[[ <Esc><Cmd>lua require('refactoring').refactor('Inline Variable')<CR>]],
			{ noremap = true, silent = true, expr = false, desc = "Refactor [i]nline variable" }
		)

		bind(
			"n",
			"<leader>rb",
			[[ <Cmd>lua require('refactoring').refactor('Extract Block')<CR>]],
			{ noremap = true, silent = true, expr = false, desc = "Refactor Extract [b]lock" }
		)
		bind(
			"n",
			"<leader>rB",
			[[ <Cmd>lua require('refactoring').refactor('Extract Block To File')<CR>]],
			{ noremap = true, silent = true, expr = false, desc = "Refactor [B]lock to File" }
		)
		bind(
			"n",
			"<leader>ri",
			[[ <Cmd>lua require('refactoring').refactor('Inline Variable')<CR>]],
			{ noremap = true, silent = true, expr = false, desc = "Refactor [i]nline variable" }
		)
	end,
}
