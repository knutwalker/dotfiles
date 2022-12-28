return {
	"ThePrimeagen/harpoon",
	lazy = true,
	init = function()
		vim.keymap.set("n", "<leader>i", function()
			require("harpoon.mark").add_file()
		end, { desc = "Add file to harpoon" })

		vim.keymap.set("n", "<leader>fi", function()
			require("harpoon.ui").toggle_quick_menu()
		end, { desc = "Toggle Harpoon [I]nterface" })

		vim.keymap.set("n", "<C-4>", function()
			require("harpoon.ui").nav_file(1)
		end, { desc = "Navigate to first file" })

		vim.keymap.set("n", "<C-5>", function()
			require("harpoon.ui").nav_file(2)
		end, { desc = "Navigate to second file" })

		vim.keymap.set("n", "<C-6>", function()
			require("harpoon.ui").nav_file(3)
		end, { desc = "Navigate to third file" })

		vim.keymap.set("n", "<C-7>", function()
			require("harpoon.ui").nav_file(4)
		end, { desc = "Navigate to fourth file" })

		vim.keymap.set("n", "<C-8>", function()
			require("harpoon.ui").nav_file(5)
		end, { desc = "Navigate to fifth file" })
	end,
}
