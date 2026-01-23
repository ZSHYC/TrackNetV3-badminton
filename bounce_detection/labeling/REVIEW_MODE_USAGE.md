# 查看模式使用说明 (Review Mode)

## 问题场景

当一个match中的所有视频都已经标注完成后，直接运行标注工具会显示：

```bash
python labeling_launcher.py --match_dir data/test/match1

Found 11 CSV files in data/test/match1\csv
Labeled: 11
Pending: 0
All files have been labeled!
```

此时无法进入查看或修改已标注的内容。

## 解决方案：使用 `--review` 标志

### 命令格式

```bash
python labeling_launcher.py --match_dir <match目录> --review
```

### 示例

```bash
# 查看已标注的match1
python labeling_launcher.py --match_dir data/test/match1 --review

# 查看已标注的match2
python labeling_launcher.py --match_dir data/test/match2 --review
```

## 功能特性

### 1. 自动加载已有标注

在查看模式下：
- ✅ 自动加载已保存的标注文件 (`*_labels.json`)
- ✅ 显示所有已确认的事件
- ✅ 不会重新运行自动检测（保留原有标注）

### 2. 视频导航保持查看模式

在查看模式下使用视频导航（`Ctrl+←` / `Ctrl+→`）时：
- ✅ 自动加载每个视频的已有标注
- ✅ 显示 `[Labeled]` 状态标记
- ✅ 无需重新标注即可快速浏览

示例输出：
```
[Switch Video] [Labeled] Loaded: match1/1_05_03 (2/11)
Loaded 32 events from data/test/match1\labels\1_05_03_labels.json
```

### 3. 支持修改和重新保存

在查看模式下，你仍然可以：
- ✅ 修改现有事件（确认、删除、更改类型）
- ✅ 添加新事件
- ✅ 保存修改（`S` 或 `Ctrl+S`）
- ✅ 切换到下一个视频继续查看

## 使用场景

### 场景1：快速查看标注效果

```bash
conda activate zsh
python labeling_launcher.py --match_dir data/test/match1 --review
```

打开后：
1. 使用 `Ctrl+→` 快速切换视频
2. 使用 `↑` / `↓` 浏览事件
3. 按 `Q` 退出

### 场景2：检查和修正错误

```bash
python labeling_launcher.py --match_dir data/test/match1 --review
```

1. 发现错误标注的事件
2. 按 `Delete` 删除错误事件
3. 按 `N` 添加缺失的事件
4. 按 `S` 保存修改
5. 使用 `Ctrl+→` 检查下一个视频

### 场景3：批量查看多个文件

启动后会逐个打开已标注文件：
```
[1/11] 1_05_02_ball.csv  ← 查看第1个
[关闭后提示]
Continue to next file? (y/n): y  ← 输入y继续
[2/11] 1_05_03_ball.csv  ← 自动打开第2个
```

## 模式对比

| 功能 | 普通模式 | 查看模式 (`--review`) |
|------|---------|---------------------|
| 处理的文件 | 未标注的文件 | 已标注的文件 |
| 自动检测 | ✅ 运行Phase 1检测 | ❌ 不运行（保留原标注） |
| 加载已有标注 | ❌ 不加载 | ✅ 自动加载 |
| 视频切换行为 | 重新检测 | 加载已有标注 |
| 可以修改 | ✅ 可以 | ✅ 可以 |
| 适用场景 | 新标注 | 查看/修改 |

## 快捷键（查看模式相同）

### 视频导航
- `Ctrl + ←` - 上一个视频（保持查看模式）
- `Ctrl + →` - 下一个视频（保持查看模式）

### 帧导航
- `←` / `A` - 上一帧
- `→` / `D` - 下一帧
- `↑` / `W` - 上一个事件
- `↓` - 下一个事件

### 编辑
- `Y` - 确认事件
- `Delete` - 删除事件
- `L` - 设置为落地
- `H` - 设置为击打
- `N` - 添加新事件

### 系统
- `S` - 保存修改
- `Q` - 退出当前视频

## 注意事项

1. **自动保存**：关闭当前视频时会自动保存修改
2. **状态显示**：UI右上角会显示 `[Labeled]` 标记
3. **继续查看**：关闭一个视频后，输入 `y` 继续查看下一个
4. **随时退出**：输入 `n` 或 `q` 停止批量查看

## 完整工作流程示例

```bash
# 1. 激活环境
conda activate zsh

# 2. 进入查看模式
python labeling_launcher.py --match_dir data/test/match1 --review

# 3. 在标注工具中
#    - 使用 Ctrl+→ 快速浏览所有视频
#    - 发现问题时修改并保存
#    - 按 Q 退出

# 4. 关闭后提示
Continue to next file? (y/n): n  # 输入n退出批量查看
```

## 提示和技巧

### 技巧1：快速预览所有视频
1. 启动查看模式
2. 只使用 `Ctrl+→` 快速切换，不做修改
3. 按 `Q` 退出

### 技巧2：集中修改某几个视频
1. 启动查看模式打开第一个视频
2. 使用 `Ctrl+→` 跳转到目标视频
3. 完成修改后按 `Q` 退出
4. 输入 `n` 不继续下一个

### 技巧3：导出前最后检查
```bash
# 1. 查看所有标注
python labeling_launcher.py --match_dir data/test/match1 --review

# 2. 确认无误后导出
python labeling_launcher.py --export data/test/match1/labels --output training_events.csv
```
