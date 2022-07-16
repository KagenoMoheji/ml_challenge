#!/bin/bash
#######################################################################################
# ●コマンド説明
# requirements.txtを基にpip installする．
# 
# ●例
# $ bash ./run_python_with_version_XX.bash -v 3.6.8 -a "-V"
# $ bash ./run_python_with_version_XX.bash -v 3.9 -a "-m pip list"
########################################################################################

com_python_version=""
com_python_args=""
while [ $# -gt 0 ]; do
    case $1 in
        -v)
            shift
            pattern_version_2d="^([0-9]+)(\.[0-9]+)(\.[0-9]+)*$"
            if [[ $1 =~ ${pattern_version_2d} ]]; then
                com_python_version="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
            fi
            ;;
        -a)
            shift
            com_python_args=$1
    esac
    shift
done

# echo ${com_python_version}

if [[ -z ${com_python_version} ]]; then
    echo "Failed in getting 2 dimensions version of Python."
    exit 2
fi

python${com_python_version} ${com_python_args}
