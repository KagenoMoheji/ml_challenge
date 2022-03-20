#!/bin/bash

#######################################################################################
# ●コマンド説明
# requirements.txtを基にpip installする．
# 
# ●例
# $ bash ./install_requirements.bash -v 3.8.6
# $ bash ./install_requirements.bash -v 3.8 -r ./conf/dependencies/user_requirements.txt
########################################################################################

com_python_version="0"
fname_requirements=""
# コマンドライン引数を取得
while [ $# -gt 0 ]; do
    case $1 in
        -v)
            shift
            pattern_version_2d="^([0-9]+)(\.[0-9]+)?(\.[0-9]+)*$"
            if [[ $1 =~ ${pattern_version_2d} ]]; then
                com_python_version="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
            fi
            ;;
        -r)
            shift
            if [ -f $1 ]; then
                fname_requirements=$1
            fi
            ;;
        *)
            ;;
    esac
    shift
done
if [ ${com_python_version} = "0" ]; then
    echo "Cannot install requirements because of not setting Python version."
    exit 1
fi
if [ -z ${fname_requirements} ]; then
    echo "Warning: Not executed 'pip install' dependencies because of not passed argument '-r'."
    exit 0
fi
# echo ${com_python_version}
# echo ${fname_requirements}



FILE="tmp_check_exist_file.txt"
ls ${fname_requirements} > $FILE
# cat $FILE
grep -E ${fname_requirements} $FILE
if [ ! $? = 0 ]; then
    # requirements.txt系が見つからなかったらpip install無しの警告出して正常終了
    echo "Warning: Not executed 'pip install' dependencies because of not found '${fname_requirements}'."
    exit 0
else
    # requirements.txt系があったらpip installする
    ## pipの更新
    python${com_python_version} -m pip install -U pip
    ## 取得したrequirements.txtでpip install
    python${com_python_version} -m pip install -r ${fname_requirements}
fi
rm $FILE