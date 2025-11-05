path="${BASH_SOURCE[0]}"
if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
    asdsip_dir_path=$(cd $(dirname $path); pwd )
    export ASDSIP_HOME_PATH="${asdsip_dir_path}"
    export LD_LIBRARY_PATH=$ASDSIP_HOME_PATH/lib:$LD_LIBRARY_PATH
    export PATH=$ASDSIP_HOME_PATH/bin:$PATH
    export ASDOPS_LOG_TO_STDOUT=0
    export ASDOPS_LOG_LEVEL=INFO
    export ASDOPS_LOG_TO_FILE=0
    export ASDOPS_LOG_TO_FILE_FLUSH=0
    export ASDOPS_LOG_TO_BOOST_TYPE=asdsip #算子库对应加速库日志类型，默认asdsip
    export ASDOPS_LOG_PATH=~
else
    echo "There is no 'set_env.sh' to import"
fi