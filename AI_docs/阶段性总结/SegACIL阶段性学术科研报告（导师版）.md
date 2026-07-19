# SegACIL 阶段性学术科研报告

阶段工作围绕 CFSSeg 二维解析增量分割链路展开，已完成基线复现、工程超参数搜索及主要候选方法筛选。Batch Size 与 RHL 维度搜索确定了稳定运行区间，但因部分历史实验的 step0 谱系不同，仅作为配置依据，不作方法增益归因。在 VOC `15-5 sequential` 上，DeepLabV3+-ResNet101 采用 `aspp_up` 作为 AIR 特征后达到 70.3565% all mIoU，较 DeepLabV3 强基线提高 0.7966 个百分点，其中新类提高 2.9188 个百分点。主线黄金重放精确复现该结果，表明架构升级和特征接口已稳定。

早期 RHL 归一化在 `none`、L2、L2-sqrt、LayerNorm 及多档 gamma 下均未获得稳定收益，原因在于静态尺度变换既未增加基函数或类别结构，也可能抹除有效幅值。现已将其重构为保留部分幅值的 PowerNorm、trace-matched gamma 与类别感知 CA-C-RLS。伪标签消融证明 fixed 0.6 在 overlap、disjoint 及 `15-1 overlap` 中有效，但在线分位阈值未超过固定阈值。`artifact_class` 相对 matched-global fixed 的六组平均优势仅 0.0020 个百分点，且两种 setting 方向不一致，故硬类别阈值路线应停止，转向将连续可靠性权重写入 C-RLS 充分统计量。

后续优先完成可靠性加权 C-RLS 的双 setting 三 seed 配对验证，并复核 BOA 正交初始化的多 seed 稳定性。随后推进 PGH-RHL-lite、PowerNorm/CA-C-RLS 因子实验。候选方法独立成立后，再迁移至 DeepLabV3+，并以 Snapshot 异构成员和 RHL-SE 2.0 检验系统级收益。
