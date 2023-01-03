return {
	"VonHeikemen/lsp-zero.nvim",
	event = "BufReadPre",
	dependencies = {
		-- LSP Support
		"neovim/nvim-lspconfig",
		"williamboman/mason.nvim",
		"williamboman/mason-lspconfig.nvim",

		-- Improved Rust support
		"simrat39/rust-tools.nvim",

		-- Formatting
		"jose-elias-alvarez/null-ls.nvim",
		"jayp0521/mason-null-ls.nvim",

		-- Autocompletion
		"hrsh7th/nvim-cmp",
		"hrsh7th/cmp-buffer",
		"hrsh7th/cmp-path",
		"hrsh7th/cmp-nvim-lsp",
		"hrsh7th/cmp-nvim-lua",
		"petertriho/cmp-git",
		"hrsh7th/cmp-emoji",
		"saadparwaiz1/cmp_luasnip",

		-- Snippets
		"L3MON4D3/LuaSnip",
		"rafamadriz/friendly-snippets",

		-- Progress Report
		"j-hui/fidget.nvim",

		-- LSP kind symbols for completion items
		"onsails/lspkind-nvim",

		-- Multiline diagnostics
		{
			url = "https://git.sr.ht/~whynothugo/lsp_lines.nvim",
		},

		-- LSP Signature
		{ "ray-x/lsp_signature.nvim", config = true },
	},
	config = function()
		local lsp = require("lsp-zero")

		lsp.preset("recommended")

		lsp.ensure_installed({
			"sumneko_lua",
			-- 'rust_analyzer',
		})

		-- Fix Undefined global 'ColorMyPencils'
		lsp.configure("sumneko_lua", {
			settings = {
				Lua = {
					diagnostics = {
						globals = { "ColorMyPencils" },
					},
				},
			},
		})

		-- setup completions
		local has_words_before = function()
			-- local unpack = unpack or table.unpack
			local line, col = unpack(vim.api.nvim_win_get_cursor(0))
			return col ~= 0 and vim.api.nvim_buf_get_lines(0, line - 1, line, true)[1]:sub(col, col):match("%s") == nil
		end

		local cmp = require("cmp")
		local snip = require("luasnip")
		local cmp_select = { behavior = cmp.SelectBehavior.Select }
		local cmp_mappings = lsp.defaults.cmp_mappings({
			-- Select previous item
			["<C-p>"] = cmp.mapping.select_prev_item(cmp_select),
			-- Select next items
			["<C-n>"] = cmp.mapping.select_next_item(cmp_select),
			-- Insert selected item
			["<C-y>"] = cmp.mapping.confirm({ select = true }),
			-- Replace with selected item
			["<CR>"] = cmp.mapping.confirm({ select = true, behavior = cmp.ConfirmBehavior.Replace }),
			-- Open usggestions
			["<C-space>"] = cmp.mapping.complete(),
			-- Close suggestions, in all modes
			["<C-e>"] = cmp.mapping.close(),
			-- Close suggestions, only in insert mode
			["<ESC>"] = cmp.mapping(function()
				cmp.close()
			end, { "i" }),
			-- Tab selects the next placeholder, the next item, expands the snippet, or calls the fallback
			["<Tab>"] = cmp.mapping(function(fallback)
				if snip.jumpable(1) then
					snip.jump(1)
				elseif cmp.visible() then
					cmp.select_next_item()
				elseif snip.expandable() then
					snip.expand()
				elseif has_words_before() then
					cmp.complete()
				else
					fallback()
				end
			end, { "i", "s" }),
			-- Shift-Tab selects the previous placeholder, the previous item, or calls the fallback
			["<S-Tab>"] = cmp.mapping(function(fallback)
				if snip.jumpable(-1) then
					snip.jump(-1)
				elseif cmp.visible() then
					cmp.select_prev_item()
				else
					fallback()
				end
			end, { "i", "s" }),
		})

		-- disable completion with tab
		-- this helps with copilot setup
		-- cmp_mappings["<Tab>"] = nil
		-- cmp_mappings["<S-Tab>"] = nil

		-- Setup LSP keymap when attached to an LSP
		-- If rust-analyzer attaches, we use the rust-tools provided actions
		local telescope = require("telescope.builtin")
		local rt = require("rust-tools")
		local buf = vim.lsp.buf
		local cl = vim.lsp.codelens

		lsp.on_attach(function(client, bufnr)
			local bind = function(keys, func, desc, mode)
				if desc then
					desc = "LSP: " .. desc
				end
				if not mode then
					mode = "n"
				end

				vim.keymap.set(mode, keys, func, { buffer = bufnr, remap = false, desc = desc })
			end

			bind("<leader>li", "<cmd>LspInfo<CR>", "Information")
			bind("<leader>lI", "<cmd>NullLsInfo<CR>", "Null-LS Information")

			local capabilities = client.server_capabilities

			if capabilities.codeActionProvider then
				if client.name == "rust_analyzer" then
					bind("<leader>R<CR>", rt.code_action_group.code_action_group, "RA Code Action", { "n", "x", "v" })
					bind([[<leader>\]], rt.hover_actions.hover_actions, "RA Hover Action", { "n", "x", "v" })
				end
				bind("<leader><CR>", buf.code_action, "Code Action", { "n", "x", "v" })
			end

			if capabilities.codeLensProvider then
				bind("<leader>lc", cl.run, "Run Code Lens")
				bind("<leader>lR", cl.refresh, "Refresh Code Lens")
			end

			if capabilities.declarationProvider then
				bind("gl", buf.declaration, "[G]o to dec[l]aration")
			end

			if capabilities.definitionProvider then
				bind("gd", telescope.lsp_definitions, "[G]o to [d]efinition")
				bind("gD", "<CMD>TroubleToggle lsp_definitions<CR>", "[G]o to [D]efinition in Trouble")
				bind("<leader>fd", telescope.lsp_definitions, "[F]ind [d]efinitions")
			end

			if capabilities.documentFormattingProvider then
				local function fmt_filter(fmt_client)
					--  only use null-ls for formatting instead of lsp server
					return fmt_client.name == "null-ls"
						or fmt_client.name == "rustfmt"
						or fmt_client.name == "rust_analyzer"
				end

				local fmt_opts = {
					format_on_save = { enabled = true },
					disabled = {},
					filter = fmt_filter,
					bufnr = bufnr,
				}

				local function format()
					buf.format(fmt_opts)
				end

				vim.api.nvim_buf_create_user_command(bufnr, "Format", format, { desc = "Format file with LSP" })

				bind("<leader>fm", format, "[F]or[m]at")
				bind("<leader>fm", format, "[F]or[m]at", "v")

				bind("<leader>fM", function()
					vim.g.autoformat_disabled = not vim.g.autoformat_disabled
					print("Autoformat: " .. vim.inspect(not vim.g.autoformat_disabled))
				end, "Toggle [F]or[m]at on save")

				local auto_group = "lsp_auto_format_" .. bufnr
				vim.api.nvim_create_augroup(auto_group, { clear = true })
				vim.api.nvim_create_autocmd("BufWritePre", {
					desc = "Auto format buffer " .. bufnr .. " before save",
					group = auto_group,
					buffer = bufnr,
					callback = function()
						if not vim.g.autoformat_disabled then
							buf.format(fmt_opts)
						end
					end,
				})
			end

			if capabilities.documentHighlightProvider then
				local highlight_name = "lsp_document_highlight_" .. bufnr
				vim.api.nvim_create_augroup(highlight_name, {})
				vim.api.nvim_create_autocmd({ "CursorHold", "CursorHoldI" }, {
					group = highlight_name,
					buffer = bufnr,
					callback = buf.document_highlight,
				})
				vim.api.nvim_create_autocmd("CursorMoved", {
					group = highlight_name,
					buffer = bufnr,
					callback = buf.clear_references,
				})
			end

			if capabilities.documentSymbolProvider then
				bind("<leader>s", telescope.lsp_document_symbols, "Search [s]ymbols in this file")
			end

			if capabilities.hoverProvider then
				bind("K", buf.hover, "Hover symbol")
				bind("<C-k>", buf.hover, "Hover symbol", "i")
				if client.name == "rust_analyzer" then
					bind("<leader>k", rt.hover_actions.hover_actions, "RA: Hover Action", { "n", "x", "v" })
				else
					bind("<leader>k", buf.hover, "Hover symbol")
				end
			end

			if capabilities.implementationProvider then
				bind("gi", telescope.lsp_implementations, "[G]o to [i]mplementation")
			end

			if capabilities.referencesProvider then
				bind("gr", telescope.lsp_references, "[G]o to [r]eferences")
				bind("gR", "<CMD>TroubleToggle lsp_references<CR>", "[G]o to [R]eferences in Trouble")
			end

			if capabilities.renameProvider then
				bind("<leader>rr", buf.rename, "[R]ename")
			end

			if capabilities.signatureHelpProvider then
				bind("<leader>h", buf.signature_help, "Signature [H]elp")
				bind("<C-h>", buf.signature_help, "Signature [H]elp", "i")
			end

			if capabilities.typeDefinitionProvider then
				bind("gy", telescope.lsp_type_definitions, "[G]o to t[y]pe definition")
				bind("gY", "<CMD>TroubleToggle lsp_type_definitions<CR>", "[G]o to t[y]pe definition in Trouble")
				bind("<leader>fy", telescope.lsp_type_definitions, "[F]ind t[y]pe definitions")
			end

			if capabilities.workspaceSymbolProvider then
				bind("<leader>S", telescope.lsp_dynamic_workspace_symbols, "Search [S]ymbols in this workspace")
				bind("<leader>fs", telescope.lsp_workspace_symbols, "Search [S]ymbols in this workspace (once)")
			end

			if client.name == "rust_analyzer" then
				local bindra = function(keys, func, desc)
					desc = "[R]ust " .. desc
					vim.keymap.set("n", keys, func, { buffer = bufnr, remap = false, desc = desc })
				end

				bindra("<leader>Rhs", rt.inlay_hints.set, "inlay [h]ints [s]et")
				bindra("<leader>Rhu", rt.inlay_hints.unset, "inlay [h]ints [u]nset")
				bindra("<leader>Rhe", rt.inlay_hints.enable, "inlay [h]ints [e]nable")
				bindra("<leader>Rhd", rt.inlay_hints.disable, "inlay [h]ints [d]isable")

				bindra("<leader>Rr", rt.runnables.runnables, "[r]unnables")
				bindra("<leader>Re", rt.expand_macro.expand_macro, "[e]xpand macro")

				bindra("<leader>Rj", function()
					rt.move_item.move_item(false)
				end, "Move item down")
				bindra("<leader>Rk", function()
					rt.move_item.move_item(true)
				end, "Move item up")

				bindra("<leader>Rha", rt.hover_actions.hover_actions, "[h]over [a]ctions")
				bindra("<leader>Rhr", rt.hover_range.hover_range, "[h]over [r]ange")

				bindra("<leader>Rc", rt.open_cargo_toml.open_cargo_toml, "[c]argo toml")
				bindra("<leader>Rp", rt.parent_module.parent_module, "[p]arent module")

				bindra("<leader>RJ", rt.join_lines.join_lines, "[J]oin lines")

				bindra("<leader>Rs", function()
					rt.ssr.ssr(vim.fn.input("SSR > "))
				end, "[s]tructural search/replace")

				bindra("<leader>RC", rt.crate_graph.view_crate_graph, "[C]rate graph")
			end
		end)

		-- Setup diagnostic keymaps
		vim.keymap.set("n", "[d", vim.diagnostic.goto_prev, { desc = "Go to previous diagnostic" })
		vim.keymap.set("n", "]d", vim.diagnostic.goto_next, { desc = "Go to next diagnostic" })
		vim.keymap.set("n", "<leader>[", vim.diagnostic.open_float, { desc = "Show diagnostic" })
		vim.keymap.set("n", "<leader>qd", vim.diagnostic.setloclist, { desc = "Set diagnostic to loclist" })

		-- Load some vscode snippets
		local snippets_path = vim.fn.stdpath("data") .. "/vscode_snippets/"
		require("luasnip.util.log").set_loglevel("info")
		require("luasnip.loaders.from_vscode").load({
			paths = {
				snippets_path .. "trusty_rusty_snippets",
				snippets_path .. "bevy-snippets",
			},
		})

		-- Actually setup completions
		local types = require("cmp.types")
		local kinds = types.lsp.CompletionItemKind
		local kind_mapping = {
			[kinds.Variable] = 1,
			[kinds.Method] = 1,
			[kinds.Function] = 1,
			[kinds.Field] = 1,
			[kinds.Property] = 1,
			[kinds.EnumMember] = 1,
			[kinds.Constant] = 1,
			[kinds.Constructor] = 1,
			[kinds.Class] = 1,
			[kinds.Enum] = 1,
			[kinds.Struct] = 1,
			[kinds.Interface] = 1,
			[kinds.TypeParameter] = 1,
			[kinds.Module] = 1,
			[kinds.Keyword] = 1,
			[kinds.Unit] = 1,
			[kinds.Value] = 1,
			[kinds.Color] = 2,
			[kinds.File] = 2,
			[kinds.Reference] = 2,
			[kinds.Folder] = 2,
			[kinds.Event] = 2,
			[kinds.Operator] = 2,
			[kinds.Snippet] = 24,
			[kinds.Text] = 25,
		}
		lsp.setup_nvim_cmp({
			mapping = cmp_mappings,
			-- Do not preselect an item from the completions
			preselect = "none",
			-- matches what is configured in set.lua
			completion = {
				completeopt = "menu,menuone,noinsert,noselect,preview",
			},
			-- configure the LSP as expansion engine, this should be set by LSP-zero as well
			snippet = {
				expand = function(args)
					snip.lsp_expand(args.body)
				end,
			},
			-- reformat the completion items with lsp_kind
			formatting = {
				format = require("lspkind").cmp_format({
					mode = "symbol_text",
					maxwidth = 50,
					ellipsis_char = "…",
					preset = "default",
					menu = {
						nvim_lsp = "[lsp]",
						nvim_lua = "[lua]",
						luasnip = "[snip]",
						crates = "[crates]",
						emoji = "[🙈]",
						git = "[git]",
						path = "[path]",
						buffer = "[buf]",
					},
				}),
			},

			-- Try to modify the order of suggested items
			sorting = {
				priority_weight = 2,
				comparators = {

					-- cmp.config.compare.exact,
					cmp.config.compare.locality,
					cmp.config.compare.recently_used,
					cmp.config.compare.sort_text,

					function(entry1, entry2)
						local kind1 = kind_mapping[entry1:get_kind()]
						local kind2 = kind_mapping[entry2:get_kind()]
						local diff = kind1 - kind2
						if diff < 0 then
							return true
						elseif diff > 0 then
							return false
						end
					end,

					cmp.config.compare.score,
					cmp.config.compare.offset,
					cmp.config.compare.order,
				},
			},

			-- setup completion sources
			sources = cmp.config.sources({
				{ name = "nvim_lsp" },
				{ name = "nvim_lua" },
				{ name = "luasnip" },
				{ name = "crates" },
				{ name = "emoji" },
				{ name = "git" },
				{ name = "path" },
			}, {
				{ name = "buffer", keyword_length = 5 },
			}),
		})

		lsp.set_preferences({
			-- use simple letters instead of fancy icons
			sign_icons = {
				error = "E",
				warn = "W",
				hint = "H",
				info = "I",
			},
			-- Don't setup default keymap, we configure our own
			set_lsp_keymaps = false,
			-- Don't configure diagnostics, we do this ourselves
			configure_diagnostics = false,
		})

		-- Setup diagnostics
		vim.diagnostic.config({
			-- Sort diagnostics by severity
			severity_sort = true,
			-- Show diagnostics in the sign column
			signs = true,
			-- Update diagnostics live while typing
			-- if false, diagnostics are only updated on InsertLeave
			update_in_insert = true,
			-- Don't show diagnostics in the virtual text
			virtual_text = false,
			-- Show diagnostics as multiple lines
			virtual_lines = true,
		})

		-- save configs for RA and null-ls since we set those up manually
		local rust_lsp_opts = lsp.build_options("rust_analyzer", {})
		local null_ls_opts = lsp.build_options("null-ls", {})

		lsp.nvim_workspace()
		lsp.setup()

		-- Setup LSP diagnostics

		-- Readme says to setup lsp-lines after lspconfig
		require("lsp_lines").setup()

		-- Toggle between multiline and single line diagnostics
		local function toggle_lsp_lines()
			local flag = not vim.diagnostic.config().virtual_lines
			print("LSP lines has been " .. (flag and "enabled" or "disabled"))
			vim.diagnostic.config({ virtual_lines = flag, virtual_text = not flag })
		end
		vim.keymap.set("n", "<leader>}", toggle_lsp_lines, { desc = "Toggle LSP line diagnostics" })
		vim.keymap.set("i", "<M-d>", toggle_lsp_lines, { desc = "Toggle LSP line diagnostics" })

		-- Use single line diagnostics while in insert mode
		-- and multiline diagnostics when not in insert mode

		vim.api.nvim_create_autocmd("InsertEnter", {
			callback = function()
				vim.diagnostic.config({ virtual_lines = false, virtual_text = true })
			end,
		})

		vim.api.nvim_create_autocmd("InsertLeave", {
			callback = function()
				vim.diagnostic.config({ virtual_lines = true, virtual_text = false })
			end,
		})

		-- Configure rust-tools

		local extension_path = vim.fn.stdpath("data") .. "/codelldb/extension/"
		local codelldb_path = extension_path .. "adapter/codelldb"
		local liblldb_path = extension_path .. "lldb/lib/liblldb.dylib"

		rust_lsp_opts.settings = {
			["rust-analyzer"] = {
				assist = {
					emitMustUse = true,
				},
				cachePriming = { numThreads = 4 },
				checkOnSave = { command = "clippy" },
				completion = { callable = { snippets = "add_parentheses" }, privateEditable = { enable = true } },
				files = { excludeDirs = { "references" } },
				hover = {
					actions = {
						debug = { enable = true },
						enable = true,
						gotoTypeDef = { enable = true },
						implementations = { enable = true },
						run = { enable = true },
					},
					documentation = {
						enable = true,
						keywords = { enable = true },
					},
					links = { enable = true },
				},
				imports = { prefix = "crate" },
				inlayHints = {
					bindingModeHints = { enable = true },
					closureReturnTypeHints = { enable = "always" },
					expressionAdjustmentHints = { enable = "reborrow" },
					lifetimeElisionHints = { enable = "skip_trivial", useParameterNames = true },
					maxLength = 42,
				},
				lens = { references = { method = { enable = true }, trait = { enable = true } } },
				procMacro = { attributes = { enable = true }, enable = true },
			},
		}

		rt.setup({
			server = rust_lsp_opts,
			tools = {
				inlay_hints = {
					only_current_line = true,
				},
				hover_actions = {
					auto_focus = true,
				},
				crate_graph = {
					backend = "dot",
				},
			},
			dap = {
				adapter = require("rust-tools.dap").get_codelldb_adapter(codelldb_path, liblldb_path),
			},
		})

		-- Configure Formatter
		local null_ls = require("null-ls")
		null_ls.setup({
			on_attach = null_ls_opts.on_attach,
			sources = {
				null_ls.builtins.diagnostics.pylint,
				null_ls.builtins.formatting.isort,
				null_ls.builtins.formatting.black,
				null_ls.builtins.formatting.stylua,
				null_ls.builtins.formatting.rustfmt,
				null_ls.builtins.code_actions.gitsigns,
			},
		})

		-- Configure git source
		require("cmp_git").setup()

		-- local mason_null_ls = require('mason-null-ls')
		-- mason_null_ls.setup({
		-- 	-- list of formatters & linters for mason to install
		-- 	ensure_installed = {
		-- 		-- 'rustmft',
		-- 		'stylua',
		-- 	},
		-- 	-- auto-install configured formatters & linters (with null-ls)
		-- 	automatic_installation = true,
		-- 	automatic_setup = false,
		-- })

		-- enable progress report
		require("fidget").setup()
	end,
}
