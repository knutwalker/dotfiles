require("lualine").setup({
	options = {
		icons_enabled = false,
		theme = "tokyonight",
		component_separators = { left = "", right = "" },
		section_separators = { left = "", right = "" },
		disabled_filetypes = {
			statusline = {},
			winbar = {},
		},
		ignore_focus = {},
		always_divide_middle = true,
		globalstatus = true,
		refresh = {
			statusline = 1000,
			tabline = 1000,
			winbar = 1000,
		},
	},
	sections = {
		lualine_a = { "mode" },
		lualine_b = { "branch", "diff", "diagnostics" },
		lualine_c = { { "filename", path = 1, file_status = true } },
		lualine_x = {
			{
				"encoding",
				separator = "",
				cond = function()
					return vim.bo.fileencoding ~= "utf-8"
				end,
			},
			{ "filetype", separator = "" },
			{
				"fileformat",
				separator = "",
				cond = function()
					return vim.bo.fileformat ~= "unix"
				end,
			},
		},
		lualine_y = { "progress" },
		lualine_z = { "location" },
	},
	inactive_sections = {
		lualine_a = {},
		lualine_b = {},
		lualine_c = { "filename" },
		lualine_x = { "location" },
		lualine_y = {},
		lualine_z = {},
	},
	tabline = {},
	winbar = {},
	inactive_winbar = {},
	extensions = { "fugitive", "fzf", "nvim-dap-ui", "quickfix" },
})
