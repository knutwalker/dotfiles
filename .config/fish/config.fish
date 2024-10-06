if status is-interactive
    # Commands to run in interactive sessions can go here
    fzf --fish | FZF_ALT_C_COMMAND= source
    bind -e \cr

    starship init fish | source

    source ~/.bash.rc/.aliases

    set fish_greeting
end
