# 交易试算 Sandbox 使用约定

## 目录结构

```
sandbox/
├── formulas/            # 融资融券公式定义，每条公式一个 .json 文件
├── scenarios/           # 业务场景 → 公式索引，每个场景一个 .json 文件
├── scratch/             # 试算中间草稿（agent 可自由读写）
├── outputs/             # 最终试算结果（agent 写入）
└── README.md            # 本文件
```

## 公式文件格式

`formulas/<formula_id>.json`：

- `formula_id`：公式唯一标识
- `name`：公式中文名
- `expression`：公式表达式
- `variables`：变量说明（对象，key 为变量名，value 为含义）
- `unit`：单位
- `notes`：口径风险 / 源表疑义等备注

## 场景文件格式

`scenarios/<scenario>.json`：

- `scenario`：场景标识
- `name`：场景中文名
- `description`：场景描述
- `formula_ids`：该场景涉及的公式 id 列表

## 使用约定

1. 涉及融资融券公式计算前，先用 `fs_grep` 或 `fs_list` 在 `formulas/` 中找到相关公式文件
2. 用 `fs_read` 读出公式文件的 `expression`，严格按文件内的表达式计算
3. 禁止自创、改写或简化公式；文件内的 `notes` 口径风险须在最终回答中显式说明
4. 试算中间结果写到 `scratch/`，最终结果写到 `outputs/`
5. 所有路径操作限定在 sandbox 根目录内，不得越权访问外部路径
