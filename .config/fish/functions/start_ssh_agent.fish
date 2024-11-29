function start_ssh_agent
    # Avoid ssh agent spawning new instances on every shell
    set ssh_agent_out $(ssh-agent -a $SSH_AUTH_SOCK -c 2>/dev/null)
    if test $status -eq 0
        eval $ssh_agent_out > /dev/null
        set -gx SSH_AUTH_SOCK $SSH_AUTH_SOCK
        set -gx SSH_AGENT_PID $SSH_AGENT_PID
    else
        set -gx SSH_AGENT_PID $(pgrep ssh-agent)
    end
end
