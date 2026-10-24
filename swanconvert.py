from pathlib import Path
from swanlab.converter import WandbLocalConverter


wandb_dir = Path.cwd() / "wandb"

run_dirs = list(wandb_dir.glob("run-*"))
latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime) if run_dirs else None

if latest_run:
    print("最新 Run 文件夹名:", latest_run.name)
    print("完整路径:", latest_run)
else:
    print("没有找到 Run 文件夹")

wb_converter = WandbLocalConverter()
# wandb_run_dir 可选
wb_converter.run(root_wandb_dir=wandb_dir, wandb_run_dir=latest_run.name)