if status is-interactive
    # Commands to run in interactive sessions can go here

    # setup path -- .path
    set -gx --prepend PATH "$HOME/.go/bin"      # Add Go installation
    set -gx --prepend PATH "/usr/local/bin"     # Add /usr/local/bin
    set -gx --prepend PATH "/usr/local/sbin"    # --''--
    set -gx --prepend PATH "/opt/homebrew/bin"  # Add aarch64 brew directories
    set -gx --prepend PATH "/opt/homebrew/sbin" # --''--
    set -gx --prepend PATH "$HOME/.cargo/bin"   # Add cargo binaries
    set -gx --prepend PATH "$HOME/.local/bin"   # Add ~/.local/bin and ~/bin
    set -gx --prepend PATH "$HOME/bin"          # --''--

    # setup env vars -- .exports

    # Make neovim the default editor.
    set -gx EDITOR 'nvim'

    # Open neovim in insert mode for git
    set -gx GIT_EDITOR 'nvim -c startinsert'

    # Make Python use UTF-8 encoding for output to stdin, stdout, and stderr.
    set -gx PYTHONIOENCODING 'UTF-8'

    # Always be able to cd from any of those directories
    set -gx --path CDPATH ".:~:~/dev:~/dev/rust:~/dev/neo"

    # Configure paths for sessionizer
    set -gx --path SESSIONIZER_PATH "~/.bash.rc:~/.config/*:~/dev/*/*"

    # Prefer US English and use UTF-8.
    set -gx LANG 'en_US.UTF-8'
    set -gx LC_ALL 'en_US.UTF-8'

    # Don’t clear the screen after quitting a manual page.
    set -gx MANPAGER 'less -X'

    # Opt-out of Homebrew analytics
    set -gx HOMEBREW_NO_ANALYTICS 1

    # Don't automatically update homebrew on every install
    set -gx HOMEBREW_NO_AUTO_UPDATE 1

    # Don't run cleanups automatically during install
    set -gx HOMEBREW_NO_INSTALL_CLEANUP 1

    # Replace the beer emoji
    set -gx HOMEBREW_INSTALL_BADGE "🍵"

    # Force cargo install-update to use default cargo over binstall
    set -gx CARGO_INSTALL_OPTS "--color auto"

    # Increase local cache size for sccache
    set -gx SCCACHE_CACHE_SIZE '100G'

    # Go PATH because ~/go is... weird
    set -gx GOPATH "$HOME/.go"

    # Avoid ssh agent spawning new instances on every invocation
    set -gx SSH_AUTH_SOCK "$HOME/.ssh/ssh-agent.sock"

    # Disable prompt manipulation by venv
    set -gx VIRTUAL_ENV_DISABLE_PROMPT 1

    # Setup FZF, see also https://owen.cymru/fzf-ripgrep-navigate-with-bash-faster-than-ever-before-2/
    set -gx FZF_DEFAULT_COMMAND 'rg --files --hidden --follow 2> /dev/null'
    set -gx FZF_CTRL_T_COMMAND "$FZF_DEFAULT_COMMAND"
    fzf --fish | FZF_ALT_C_COMMAND= source
    bind -e \cr

    # Setup prompt -- .prompt
    starship init fish | source

    # Setup aliases -- .aliases
    source ~/.bash.rc/.aliases

    # Disable greeting message
    set fish_greeting
end
