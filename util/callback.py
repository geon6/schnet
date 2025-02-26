from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
)
from util.config import conf


def get_pl_callbacks():
    c = conf.callback
    callbacks = []
    if c.early_stopping.enable:
        early_stop_callback = EarlyStopping(
            monitor = c.early_stopping.monitor, # 监控的指标
            min_delta=c.early_stopping.min_delta, # 最小变化
            patience=c.early_stopping.patience, 
            verbose=True,  # 输出详细信息
            mode=c.early_stopping.mode  # 尝试最小化监控指标
        )
        callbacks.append(early_stop_callback)

    if c.checkpoint.enable:
        checkpoint_callback = ModelCheckpoint(
            monitor=c.checkpoint.monitor,  # 监控的指标
            dirpath=c.checkpoint.dirpath,  # 检查点保存路径
            filename='{epoch:03d}-{val_mae:.4f}',  # 文件名格式
            save_top_k=c.checkpoint.save_top_k,  # 保存最佳的k个模型
            mode=c.checkpoint.mode  # 对于监控指标是寻找最小值还是最大值
        )
        callbacks.append(checkpoint_callback)

    if c.summary.enable:
        rich_model_summary_callback = RichModelSummary(max_depth=c.summary.max_depth)
        callbacks.append(rich_model_summary_callback)

    return callbacks
