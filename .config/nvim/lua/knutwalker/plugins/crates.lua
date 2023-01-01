return {
	"saecki/crates.nvim",
	event = { "BufRead Cargo.toml" },
	requires = { { "nvim-lua/plenary.nvim" } },
	config = function()
		local crates = require("crates")
		crates.setup()

		local function bind(key, cmd, desc, mode)
			mode = mode or "n"
			vim.keymap.set(mode, key, cmd, { desc = desc, noremap = true, silent = true })
		end

		bind("<leader>ct", crates.toggle, "[t]oggle [c]rate UI elements")
		bind("<leader>cr", crates.reload, "[r]eload [c]rate data")

		bind("<leader>ck", crates.show_popup, "Show [c]rates popup")
		bind("<leader>cK", crates.show_popup, "Show [c]rates popup")
		bind("<leader>cc", crates.show_crate_popup, "Show [c]rate details popup")
		bind("<leader>cv", crates.show_versions_popup, "Show [c]rates [v]ersions popup")
		bind("<leader>cf", crates.show_features_popup, "Show [c]rates [f]eatures popup")
		bind("<leader>cd", crates.show_dependencies_popup, "Show [c]rates [d]ependencies popup")

		bind("<leader>cu", crates.update_crate, "[u]pdate [c]rate on current line")
		bind("<leader>cu", crates.update_crate, "[u]pate [c]rates in selection", "v")
		bind("<leader>ca", crates.update_all_crates, "Update [a]ll [c]rates")
		bind("<leader>cU", crates.upgrade_crate, "[U]pgrade [c]rate on current line")
		bind("<leader>cU", crates.upgrade_crate, "[U]pgrade [c]rates in selection", "v")
		bind("<leader>cA", crates.upgrade_all_crates, "Upgrade [A]ll [c]rates")

		bind("<leader>cH", crates.open_homepage, "Open [c]rate [H]omepage")
		bind("<leader>cR", crates.open_repository, "Dpen [c]rate [R]epository")
		bind("<leader>cD", crates.open_documentation, "Open [c]rate [D]ocumentation")
		bind("<leader>cC", crates.open_crates_io, "Open [c]rates.io")
	end,
}
