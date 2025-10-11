path="${BASH_SOURCE[0]}"
if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
    asdsip_dir_path=$(cd "$(dirname "$path")"; pwd )
    export ASDSIP_HOME_PATH="${asdsip_dir_path}/latest/"
    export LD_LIBRARY_PATH=$ASDSIP_HOME_PATH/lib:$LD_LIBRARY_PATH
else
    echo "There is no 'set_env.sh' to import"
fi
