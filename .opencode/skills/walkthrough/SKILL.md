---
name: Daily Walkthrough
description: Record the day's repository changes in doc/YYYY-MM/YYYY-MM-DD.md, then commit and push. Use when the user asks for a daily walkthrough, daily log, daily change record, 当天修改记录, 更新 walkthrough, or 记录并推送今天的修改.
---

# Daily Walkthrough

## 何时使用

当用户要求以下内容时使用本 skill：

- 记录今天改了什么
- 更新 daily walkthrough / 日志 / 日报
- 按日期归档修改
- 记录并 git push

## 记录范围

只记录**当前项目文件夹（本仓库根目录）内**的修改：

- 以 `git rev-parse --show-toplevel` 得到的项目根目录为边界。
- 只查看和记录该目录下的 `git status` / `git diff` / 文件改动。
- 不记录仓库外、其他项目、全局配置、训练日志、`wandb/`、模型 checkpoint 等。
- 如果当前项目只是更大仓库的子目录，只记录当前子目录内的路径，不要记录子目录外的改动。

Git 查询命令要带路径限定，例如：

```bash
git status --short -- .
git diff --stat -- .
git diff -- .
```

## 目录约定

以仓库根目录为基准：

```text
doc/
└── YYYY-MM/
    └── YYYY-MM-DD.md
```

- 月份文件夹：本地日期的 `YYYY-MM`
- 每日文件：本地日期的 `YYYY-MM-DD.md`
- 一天一个文件
- 同一天有多次工作时，在文件末尾追加 `## HH:MM 追加记录`，不要覆盖已有内容

## 工作流

1. 确认仓库根目录：
   ```bash
   git rev-parse --show-toplevel
   ```
2. 取本地日期：
   ```bash
   date +%Y-%m
   date +%F
   date +%H:%M
   ```
3. 收集实际改动：
   ```bash
   git status --short
   git diff --stat
   git diff
   git log -1 --oneline
   ```
   对未跟踪文件要单独检查，不要未经确认就把大文件或生成物加入提交。

4. 创建或更新当天文件：
   ```bash
   mkdir -p doc/YYYY-MM
   ```
   如果文件已存在，只追加，不要重写历史内容。

5. 按下面的模板记录。内容用中文，重点写：
   - 为什么改
   - 改了哪些文件
   - 关键实现/参数
   - 如何验证
   - 训练/实验 run id 和观察结果
   - 未完成事项

6. 检查暂存内容：
   ```bash
   git diff --cached --stat
   ```

7. 提交并推送：
   ```bash
   git commit -m "daily: YYYY-MM-DD <简短摘要>"
   git push origin "$(git branch --show-current)"
   ```

8. 如果今天没有实际改动，不要创建空提交，直接说明没有需要记录的变化。

## 文档模板

````markdown
# YYYY-MM-DD Walkthrough

## 今日目标

- ...

## 修改内容

### 1. <主题>

- 文件：`path/to/file.py`
- 改动：
  - ...
- 关键参数：
  ```text
  ...
  ```
- 验证：
  ```text
  ...
  ```

## 训练 / 实验

- run：
- 配置：
- 结果：
- 结论：

## 验证记录

```bash
# 执行过的关键命令
```

## 待办

- ...

## 备注

- ...
````

同一天追加时使用：

```markdown
## HH:MM 追加记录

### <主题>

- ...
```

## Git 提交规则

默认提交：

- 当天的 walkthrough 文档
- 本 skill 的修改
- 文档中记录到的、且位于当前项目文件夹内的当天源码/配置/脚本改动

不要 stage 或 commit 当前项目文件夹之外的任何文件。

默认不要提交：

- `MUJOCO_LOG.TXT`
- `myrobot.tar.gz` 等大压缩包
- `wandb/`、`logs/`、`swanlog/`
- `*.pt`、`*.onnx`、模型 checkpoint
- `.venv/`、`__pycache__/`
- 临时文件、空文件、用途不明的未跟踪文件

如果存在以下情况，先询问用户：

- 有大量未跟踪文件
- 有删除操作
- 有看起来与今天工作无关的改动
- 需要 force push、改写历史、跨分支 push

提交前必须确认：

```bash
git status --short
git diff --cached --stat
```

推送失败时不要强推。先检查远端是否有新提交，再决定是否 `git pull --rebase`。

## 质量要求

- 只记录真实发生的修改，不要编造
- 写明文件路径和关键参数
- 记录训练 run 的目录名/checkpoint/关键指标
- 明确区分“已验证”和“待验证”
- 文档保持可读，不要粘贴大段无关日志
