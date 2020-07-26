#!/bin/bash

# (partially taken from: https://github.com/cykerway/complete-alias)

_use_alias=1

# Disable the use of alias for a command.
_disable_alias () {
    local cmd="$1"

    # Remove completion for this command.
    complete -r "$cmd"

    # Reset static completions.
    #
    # We don't know the original no-alias completion for $cmd because it has
    # been overwritten by the alias completion function. What we do here is that
    # we reset all static completions to those in vanilla bash_completion. This
    # may be an overkill becase we only need to reset completion for $cmd, but
    # it works.
    complete -u groups slay w sux
    complete -A stopped -P '"%' -S '"' bg
    complete -j -P '"%' -S '"' fg jobs disown
    complete -v readonly unset
    complete -A setopt set
    complete -A shopt shopt
    complete -A helptopic help
    complete -a unalias
    complete -A binding bind
    complete -c command type which
    complete -b builtin
    complete -F _service service
    complete -F _known_hosts traceroute traceroute6 tracepath tracepath6 \
        fping fping6 telnet rsh rlogin ftp dig mtr ssh-installkeys showmount
    complete -F _command aoss command do else eval exec ltrace nice nohup \
        padsp then time tsocks vsound xargs
    complete -F _root_command fakeroot gksu gksudo kdesudo really
    complete -F _longopt a2ps awk base64 bash bc bison cat chroot colordiff cp \
        csplit cut date df diff dir du enscript env expand fmt fold gperf \
        grep grub head irb ld ldd less ln ls m4 md5sum mkdir mkfifo mknod \
        mv netstat nl nm objcopy objdump od paste pr ptx readelf rm rmdir \
        sed seq sha{,1,224,256,384,512}sum shar sort split strip sum tac tail tee \
        texindex touch tr uname unexpand uniq units vdir wc who
    complete -F _minimal ''
    complete -D -F _completion_loader

    # Reset _use_alias flag.
    _use_alias=0
}

# Enable the use of alias for a command.
_enable_alias () {
    local cmd="$1"

    # Set completion for this command.
    complete -F _complete_alias "$cmd"

    # Set _use_alias flag.
    _use_alias=1
}

# Expand the first command as an alias, stripping all leading redirections.
_expand_alias () {
    local alias_name="${COMP_WORDS[0]}"
    local alias_namelen="${#alias_name}"
    local alias_array=( $(alias "$alias_name" | sed -r 's/[^=]*=//' | xargs) )
    local alias_arraylen="${#alias_array[@]}"
    local alias_str="${alias_array[*]}"
    local alias_strlen="${#alias_str}"

    # Rewrite current completion context by expanding alias.
    COMP_WORDS=(${alias_array[@]} ${COMP_WORDS[@]:1})
    (( COMP_CWORD+=($alias_arraylen-1) ))
    COMP_LINE="$alias_str""${COMP_LINE:$alias_namelen}"
    (( COMP_POINT+=($alias_strlen-$alias_namelen) ))

    # Strip leading redirections in alias-expanded command line.
    local redir="@(?([0-9])<|?([0-9&])>?(>)|>&)"
    while [[ "${#COMP_WORDS[@]}" -gt 0 && "${COMP_WORDS[0]}" == $redir* ]]; do
        local word="${COMP_WORDS[0]}"
        COMP_WORDS=(${COMP_WORDS[@]:1})
        (( COMP_CWORD-- ))
        local linelen="${#COMP_LINE}"
        COMP_LINE="${COMP_LINE#$word+( )}"
        (( COMP_POINT-=($linelen-${#COMP_LINE}) ))
    done
}

# alias completion function.
_complete_alias () {
    local cmd="${COMP_WORDS[0]}"

    if [[ "$_use_alias" -eq 1 ]]; then
        _expand_alias
    fi
    _disable_alias "$cmd"
    _command_offset 0
    _enable_alias "$cmd"
}


# System utils aliases

alias x='xdg-open'

alias b='byobu'

alias tl='trash-list'
alias te='trash-empty'
alias t='trash'
complete -F _complete_alias t

alias duh='clr; pwd; du -had1 | sort -hr'

alias clr='tput reset' # like clear, but actually clears the terminal, erasing previous output

alias S='source ~/.bashrc'

alias sai='sudo apt install'
complete -F _complete_alias sai

alias s='clr; git s'
complete -F _complete_alias s

alias push='git push'
complete -F _complete_alias push

alias cap='pygmentize -g' # replacement of cat with python-pygments to cat with colors

alias ipy='ipython3'
complete -F _complete_alias ipy

# ROS1 aliases

alias r='rviz'

alias rc='roscore'

alias cb='catkin build -c'
complete -F _complete_alias cb

alias cbt='catkin build --this'
complete -F _complete_alias cbt

alias rt='rostopic'
complete -F _complete_alias rt

alias rs='rosservice'
complete -F _complete_alias rs

alias rn='rosnode'
complete -F _complete_alias rn

alias rl='roslaunch'
complete -F _complete_alias rl

alias rr='rosrun'
complete -F _complete_alias rr

alias rb='rosbag'
complete -F _complete_alias rb

alias rp='rospkg'
complete -F _complete_alias rp

alias rcd='roscd'
complete -F _complete_alias rcd

alias ust='rosparam set use_sim_time true'

