一些点子:
1. 能不能预测Rfam里的核酸序列的二级结构，然后类似在某个数据库里搜索msa的思路在这些二级结构里搜索相似的二级结构

2. 在做crop的时候，考虑二级结构


model：
embedders.py（还差recycle）
evoformer.py（还差BUG和ExtraMSAStack）


utils已阅：
checkpointing.py, chunk_utils.py


不同attention的使用条件：
如果启用FP16混合精度，则禁用memory_efficient_kernel
Flash Attention 可以处理带有 mask 的注意力机制，但不直接支持带有 pair bias 的 attention。pair bias 是指可以直接添加到注意力权重上的偏置项，而 mask 通常需要经过额外的变换步骤才能应用到注意力权重中。Flash Attention 内部已经包含了处理 mask 的逻辑，因此可以高效地处理 mask
memory_efficient_kernel和deepspeed_evo_attention均最多支持两个bias（即pair bias和mask）


弃用memory_efficient_kernel，及其相关的attn_core_inplace_cuda


todo: 

可以先想一下训练和预测过程中未知的残基该怎么办


对于训练集里的样本，可能存在未知残基，以及已知残基的某些原子缺失，这些地方用pad填充
对于验证集和测试集里的样本，未知残基要转化为已知残基

可以开始写数据集和数据处理了
关于序列中的unk还有一个点，我是使用mmcif_parsing的chain_to_seqres直接几乎作为最终序列（AUCG以外的要转化为X），
没有复杂的额外步骤将一些其他残基转化为已知的。后面也都要遵循这一点，包括从序列出发的东西。

alphafold在structure module中的每一个layer中额外计算辅助损失：backbone的FAPE和扭转角的loss
预测的frames需要扭转角，从而获得全原子坐标。所以我猜测，alphafold需要在每个layer中直接优化扭转角，而不是选择
在最后计算扭转角损失。
因此，迁移到rna的项目中，我只有距离图这种pair的损失，且不是直接影响原子坐标，所以还是选择放最后计算损失

heads处理绝大部分了，model才刚开始。
先写数据处理


训练集：对于未知残基，其all_atom_mask全为0（三个原子均为0），对于已知残基的某些原子缺失，缺失的部分mask为0
此时，未知残基在处理上直接等价于disorder区域
