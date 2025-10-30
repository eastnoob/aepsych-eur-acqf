# Git 仓库快速参考

## 仓库位置

```
d:\WORKSPACE\python\aepsych-source\extensions\dynamic_eur_acquisition\.git
```

## 基本操作

### 查看状态

```bash
cd d:\WORKSPACE\python\aepsych-source\extensions\dynamic_eur_acquisition
git status
```

### 查看历史

```bash
# 简洁版本
git log --oneline --all --graph

# 详细版本
git log --stat
```

### 查看标签

```bash
# 列出所有标签
git tag

# 查看标签详情
git show v1.0-v2.0-experiment
```

### 查看差异

```bash
# 查看工作区变化
git diff

# 查看已暂存的变化
git diff --cached

# 查看两个提交之间的差异
git diff <commit1> <commit2>
```

## 提交工作流

### 添加文件

```bash
# 添加单个文件
git add <file>

# 添加所有变更
git add -A

# 添加当前目录下的所有变更
git add .
```

### 提交变更

```bash
# 提交并附带消息
git commit -m "Your commit message"

# 提交并打开编辑器写详细消息
git commit
```

### 修改最后一次提交

```bash
# 修改提交消息
git commit --amend -m "New message"

# 添加遗漏的文件到最后一次提交
git add <forgotten-file>
git commit --amend --no-edit
```

## 分支操作

### 查看分支

```bash
# 列出本地分支
git branch

# 列出所有分支（包括远程）
git branch -a
```

### 创建和切换分支

```bash
# 创建新分支
git branch <branch-name>

# 切换到分支
git checkout <branch-name>

# 创建并切换到新分支（推荐）
git checkout -b <branch-name>
```

### 合并分支

```bash
# 切换到目标分支
git checkout master

# 合并其他分支
git merge <branch-name>
```

## 标签操作

### 创建标签

```bash
# 轻量标签
git tag <tag-name>

# 注释标签（推荐）
git tag -a <tag-name> -m "Tag description"
```

### 删除标签

```bash
git tag -d <tag-name>
```

## 回退和撤销

### 撤销工作区的修改

```bash
# 撤销单个文件
git checkout -- <file>

# 撤销所有修改
git checkout .
```

### 撤销已暂存的修改

```bash
# 取消暂存单个文件
git reset HEAD <file>

# 取消所有暂存
git reset HEAD
```

### 回退到特定提交

```bash
# 软回退（保留工作区和暂存区）
git reset --soft <commit>

# 混合回退（保留工作区，清空暂存区）
git reset --mixed <commit>

# 硬回退（清空工作区和暂存区）⚠️ 谨慎使用
git reset --hard <commit>
```

### 查看特定版本的文件

```bash
# 查看文件内容
git show <commit>:<file-path>

# 恢复文件到特定版本
git checkout <commit> -- <file-path>
```

## 远程仓库操作

### 添加远程仓库

```bash
# 添加远程仓库
git remote add origin <url>

# 查看远程仓库
git remote -v
```

### 推送到远程

```bash
# 首次推送并设置上游
git push -u origin master

# 后续推送
git push

# 推送标签
git push origin --tags
```

### 从远程拉取

```bash
# 拉取并合并
git pull

# 仅拉取不合并
git fetch
```

## 常用场景

### 场景 1: 开始开发 V3

```bash
# 创建 V3 开发分支
git checkout -b v3-development

# 修改代码...
# 测试...

# 提交变更
git add -A
git commit -m "Implement V3: V1 + hard exclusion"

# 创建标签
git tag -a v3.0 -m "V3: Minimal improvement"

# 合并回 master
git checkout master
git merge v3-development
```

### 场景 2: 回到 V1+V2 实验状态

```bash
# 查看标签
git tag

# 检出到标签状态
git checkout v1.0-v2.0-experiment

# 查看代码...

# 返回最新状态
git checkout master
```

### 场景 3: 对比 V1 和 V2

```bash
# 查看 V2 相对于 V1 的变更
git diff v1.0-v2.0-experiment HEAD -- acquisition_function_v2.py

# 查看特定文件的历史
git log --follow -- acquisition_function_v2.py
```

### 场景 4: 保存当前工作但不提交

```bash
# 暂存当前工作
git stash

# 切换到其他分支工作...

# 恢复暂存的工作
git stash pop
```

## 文件忽略

当前 `.gitignore` 配置忽略：

- `__pycache__/` - Python 缓存
- `*.pyc`, `*.pyo` - 编译文件
- `venv/`, `env/` - 虚拟环境
- `*.db`, `*.sqlite` - 数据库
- `results/`, `results_v2/` - 实验结果
- `.vscode/`, `.idea/` - IDE 配置

如需追踪被忽略的文件：

```bash
# 强制添加
git add -f <ignored-file>

# 或修改 .gitignore
```

## 实用别名（可选）

在仓库中设置 Git 别名：

```bash
git config alias.st status
git config alias.co checkout
git config alias.br branch
git config alias.ci commit
git config alias.unstage 'reset HEAD --'
git config alias.last 'log -1 HEAD'
git config alias.visual 'log --oneline --all --graph --decorate'
```

使用别名：

```bash
git st        # 相当于 git status
git visual    # 查看图形化历史
```

## 当前仓库状态

- **分支**: master
- **最新提交**: 6093769 "Add Git repository creation report"
- **标签**: v1.0-v2.0-experiment (提交 d0b3228)
- **文件数**: 69
- **状态**: Clean (无未提交变更)

## 重要提示

### ✅ 安全操作

- 定期提交：`git add -A && git commit -m "Descriptive message"`
- 使用分支：`git checkout -b feature-name`
- 查看状态：`git status`
- 查看历史：`git log --oneline`

### ⚠️ 危险操作

- `git reset --hard` - 会永久删除未提交的更改
- `git push -f` - 强制推送会覆盖远程历史
- 删除 `.git` 目录 - 会丢失所有历史

### 💡 最佳实践

1. 小而频繁的提交 - 每个提交解决一个明确的问题
2. 清晰的提交消息 - 描述做了什么和为什么
3. 使用分支开发 - 保持 master 分支稳定
4. 创建标签标记里程碑 - 便于版本管理
5. 定期查看状态 - 了解工作区状态

## 快速检查清单

在进行重要操作前：

- [ ] `git status` - 确认工作区状态
- [ ] `git diff` - 查看具体变更
- [ ] `git log` - 了解历史
- [ ] 创建备份分支 - `git branch backup-YYYYMMDD`

---

更新时间: 2025年10月30日
当前版本: v1.0-v2.0-experiment
下一步: 开发 V3 或进行其他实验
