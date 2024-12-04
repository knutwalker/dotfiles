
if status is-interactive
    # Commands to run in interactive sessions can go here

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
