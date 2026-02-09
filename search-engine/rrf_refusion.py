
import numpy as np

def rrf_fusion(multi_source_data, k=60, reliable_source_idx=None, weight=1.2):
    """
    RRF融合算法Python实现
    :param multi_source_data: 多数据源数据，格式为列表，每个元素为字典{目标: 排序rank（从1开始）}
    :param k: 平滑参数，默认60，可根据数据源数量调整
    :param reliable_source_idx: 可靠数据源的索引（如[0]表示第一个数据源更可靠），默认None
    :param weight: 可靠数据源的权重系数，默认1.2
    :return: 融合后的结果，字典{目标: 融合得分}，按得分降序排列
    """
    # 第一步：数据预处理（对齐目标、处理异常值）
    # 1. 提取所有数据源的目标集合，取交集（对齐目标）
    all_targets = set(multi_source_data[0].keys())
    for data in multi_source_data[1:]:
        all_targets.intersection_update(set(data.keys()))
    all_targets = list(all_targets)  # 所有数据源的交集目标，避免无效目标干扰
    
    # 2. 处理排序异常值（空值、负值），统一替换为最大排序值+1
    processed_data = []
    for data in multi_source_data:
        max_rank = max(data.values()) if data else 1
        processed = {}
        for target in all_targets:
            rank = data.get(target, max_rank + 1)
            # 处理异常rank（确保从1开始）
            if rank is None or rank < 1:
                rank = max_rank + 1
            processed[target] = rank
        processed_data.append(processed)
    
    # 第二步：参数配置（默认k=60，可手动调整，贴合前文参数设定）
    # 第三步：得分计算与融合（支持可靠数据源加权，贴合前文实操要点）
    fusion_scores = {}
    # 向量化计算优化（替代循环，提升效率，适配大量目标场景）
    targets_arr = np.array(all_targets)
    # 提取每个目标在各数据源的rank值，生成矩阵（行数=目标数，列数=数据源数）
    rank_matrix = np.array([[processed_data[i][t] for i in range(len(processed_data))] for t in all_targets])
    # 计算每个目标在各数据源的贡献得分：1/(rank + k)
    contribution_matrix = 1 / (rank_matrix + k)
    
    # 给可靠数据源加权（若有）
    if reliable_source_idx is not None:
        # 初始化权重矩阵（默认权重1.0）
        weight_matrix = np.ones_like(contribution_matrix)
        # 给可靠数据源设置指定权重
        for idx in reliable_source_idx:
            weight_matrix[:, idx] = weight
        # 加权计算贡献得分
        contribution_matrix = contribution_matrix * weight_matrix
    
    # 求和得到每个目标的最终融合得分
    fusion_scores_arr = np.sum(contribution_matrix, axis=1)
    
    # 拼接目标与对应融合得分，生成字典
    fusion_scores = dict(zip(targets_arr, fusion_scores_arr))
    
    # 第四步：结果生成（按融合得分降序排序）
    sorted_fusion_result = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_fusion_result

# ------------------- 示例：多模型目标检测结果融合 -------------------
if __name__ == "__main__":
    # 模拟3个数据源（比如3个目标检测模型的输出，key=目标ID，value=排序rank）
    source1 = {"target1": 1, "target2": 3, "target3": 2, "target4": 4}  # 模型1排序
    source2 = {"target1": 2, "target2": 1, "target3": 4, "target4": 3}  # 模型2排序
    source3 = {"target1": 3, "target2": 2, "target3": 1, "target4": 4}  # 模型3排序（假设更可靠）
    multi_source = [source1, source2, source3]
    
    # 调用RRF融合函数（指定source3为可靠数据源，索引为2，权重1.2）
    fusion_result = rrf_fusion(
        multi_source_data=multi_source,
        k=60,  # 平滑参数，贴合默认值
        reliable_source_idx=[2],  # 可靠数据源索引
        weight=1.2
    )
    
    # 输出融合结果
    print("RRF融合后的结果（按得分降序）：")
    for target, score in fusion_result:
        print(f"目标{target}: 融合得分 = {score:.4f}")
