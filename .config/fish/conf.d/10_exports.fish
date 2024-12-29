# Make neovim the default editor.
set -gx EDITOR 'nvim'

# Open neovim in insert mode for git
set -gx GIT_EDITOR 'nvim -c startinsert'

# Make Python use UTF-8 encoding for output to stdin, stdout, and stderr.
set -gx PYTHONIOENCODING 'UTF-8'

# Always be able to cd from any of those directories
set -gx --path CDPATH ".:~:~/dev:~/dev/rust:~/dev/zig"

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

# Set xdg vars since some apps prefer that over the mac defaults
set -gx XDG_CONFIG_HOME "$HOME/.config"
set -gx XDG_CACHE_HOME "$HOME/.cache"
set -gx XDG_DATA_HOME "$HOME/.local/share"
set -gx XDG_STATE_HOME "$HOME/.local/state"
