# AGENTS.md

## Scope
这是/home/xy.chen/tw/ECO-humanoid-remote/eco-humanoid-energy-experiments项目的开发规范，主要针对 Isaac Gym（NVIDIA 物理仿真平台）相关的训练、评估和调试代码。
python环境为conda里的bruce_gym

## Language

- 默认使用中文回复，并称呼用户为 `nullptr`。
- English terms must include Chinese meaning on first use, for example GitHub（代码托管平台）, Git（版本控制系统）, branch（分支）, commit（提交）, and push（推送）。
- 不确定时必须明确说明不确定点，不要编造不存在的文件、接口、命令或结果。
- 生成的各种数据文件，png，csv，md等，尽量使用中文描述，如是英语专业名称，旁边需要附上中文解释。

## Environment
- 本机是 Ubuntu 22.04服务器，使用 NVIDIA GPU 进行 Isaac Gym 训练和评估。无GUI环境，使用 SSH 远程连接。
- 除非我主动要求，否则不要用命令行帮我运行需要gpu的代码，所有需要gpu训练的代码运行必须由我本人在本机终端执行。只需要教我怎么使用命令行运行，我会在终端复制粘贴。
- 对于不需要gpu训练的代码，可以在本机终端直接运行。例如查看log日志，ps等代码直接运行。
- 给出的命令行尽量不要用export，选定显卡应该在命令行中直接指定CUDA_VISIBLE_DEVICES，而不是在~/.bashrc或~/.zshrc中设置。
- 命令行有修改变动，应该跟我说明改动会影响什么，特别是会不会影响训练出来的模型权重。
- 该服务器上的gpu5无法使用。gpu0和gpu1是A100。gpu2-4是V100。

## GitHub Sync Requirement

- 每次对代码进行改动后，必须把改动上传到 GitHub（代码托管平台）。
- 上传到 GitHub 必须经过 Git（版本控制系统）的完整流程：检查当前 branch（分支）、暂存目标文件、创建 commit（提交）、执行 push（推送）。
- 不得把“已保存到本地”“已暂存”或“已提交”说成“已上传 GitHub”。只有 push 成功后，才能说改动已上传到 GitHub。
- 如果 push 因网络、权限、认证或远端冲突失败，必须说明失败原因、当前本地 commit 哈希，以及远端尚未收到该改动。

## Branch Rules

- branch（分支）表示一条独立开发线，不等同于 commit（提交）。
- 开始改代码前先检查当前分支，并确认改动应落在哪个分支。
- 默认推送到当前分支的 upstream（上游分支）。如果没有 upstream，先说明将要推送到哪个远端分支。
- 不同任务应优先使用不同分支，避免把互不相关的代码改动混在同一分支。
- 切换或新建分支前，必须说明原因，并保护已有未提交改动。

## Commit Rules

- commit（提交）表示一次可追踪的代码快照，不等同于 branch（分支）。
- 每个 commit 应只包含一个清晰目的的改动，避免混入无关文件、临时输出、日志、缓存或大体积评估结果。
- 提交信息要简洁说明行为和目的，例如 `Fix checkpoint path resolution`。
- 对已有未提交改动要先识别来源。不要擅自提交用户的无关改动；只提交本次任务明确涉及的文件。
- 提交前运行与改动风险匹配的检查或测试。无法运行时，必须说明原因。

## Push Reporting

- push（推送）成功后，必须报告：
  - 当前 branch（分支）名称。
  - 最新 commit（提交）哈希和提交信息。
  - 推送到的 GitHub 远端分支。
  - 已运行的检查或测试。
- 如果工作区仍有未提交改动，必须说明这些改动未包含在刚刚的 commit 中。

## Generated Files

- 不要默认提交生成的评估输出、日志、缓存、模型权重或大体积数据文件。
- 只有用户明确要求保留这些文件到版本库时，才可以把它们纳入 commit（提交）。
