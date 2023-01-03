return {
	"monaqa/dial.nvim",
	init = function()
		vim.keymap.set("n", "<C-a>", function()
			return require("dial.map").inc_normal()
		end, { expr = true, noremap = true })

		vim.keymap.set("n", "<C-x>", function()
			return require("dial.map").dec_normal()
		end, { expr = true, noremap = true })

		vim.keymap.set("v", "<C-a>", function()
			return require("dial.map").inc_visual()
		end, { expr = true, noremap = true })

		vim.keymap.set("v", "<C-x>", function()
			return require("dial.map").dec_visual()
		end, { expr = true, noremap = true })

		vim.keymap.set("v", "g<C-a>", function()
			return require("dial.map").inc_gvisual()
		end, { expr = true, noremap = true })

		vim.keymap.set("v", "g<C-x>", function()
			return require("dial.map").dec_gvisual()
		end, { expr = true, noremap = true })
	end,
	config = function()
		local augend = require("dial.augend")
		require("dial.config").augends:register_group({
			default = {
				augend.integer.alias.decimal,
				augend.integer.alias.hex,
				augend.date.alias["%Y/%m/%d"],
				augend.date.alias["%Y-%m-%d"],
				augend.date.alias["%H:%M"],
				augend.date.new({
					pattern = "%a",
					default_kind = "day",
					word = true,
				}),
				augend.date.new({
					pattern = "%A",
					default_kind = "day",
					word = true,
				}),
				augend.date.new({
					pattern = "%b",
					default_kind = "day",
					word = true,
				}),
				augend.date.new({
					pattern = "%B",
					default_kind = "day",
					word = true,
				}),
				augend.constant.alias.bool,
				augend.semver.alias.semver,
			},
		})
	end,
}
