# main历史启动脚本

这里保存整合前`main@47a512a`的run.sh、run_trs.sh、run_rhl_norm.sh和run_origin.sh原文。

旧TRS脚本固定V3与历史输出目录，RHL脚本单独维护训练参数，run_origin包含旧机器路径。当前使用根目录run.sh及其两个薄封装入口，模型、数据路径、step0来源与验证协议均由参数控制。历史脚本用于对照过去的实验配方。
