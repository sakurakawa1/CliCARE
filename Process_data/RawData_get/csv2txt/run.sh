#!/bin/bash

# 处理模块名称（不带.py/.txt后缀）
modules=(
    "admissions"
    "chartevents"
    "diagnoses_icd"
    "drgcodes"
    "emar"
    "icustays"
    "ingredientevents"
    "inputevents"
    "labevents"
    "microbiologyevents"
    "outputevents"
    "prescriptions"
    "procedureevents"
    "procedures_icd"
    "transfers"
    "emar_detail"
)

# 遍历模块执行处理
for module in "${modules[@]}"; do
    py_script="${module}.py"
    out_file="${module}.txt"

    echo ">>> 正在处理: ${module}"

    if [ -f "$py_script" ]; then
        python -u "$py_script" > "$out_file"
        if [ $? -eq 0 ]; then
            cat "$out_file" | python -u local.py "${module}"
        else
            echo "[ERROR] 执行 $py_script 失败！"
        fi
    else
        echo "[WARN] 脚本不存在: $py_script"
    fi

    echo ">>> 处理完成: ${module}"
    echo
done

echo ">>> 所有处理已完成。"
