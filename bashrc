
[ -z "$PS1" ] && return

HISTCONTROL=ignoredups:ignorespace

shopt -s histappend

HISTSIZE=1000
HISTFILESIZE=2000

shopt -s checkwinsize

[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

if [ -z "$debian_chroot" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '

alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias gs='git status'
alias gl='git log'
alias gc='git checkout'
alias gcm='git commit -m'
alias push='git push'
alias pull='git pull'
alias br='source ~/.bashrc'
alias nv='watch -n 0.1 nvidia-smi'

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

source /etc/profile
source /etc/autodl-motd
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1