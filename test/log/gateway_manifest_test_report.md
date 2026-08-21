# Gateway 与 Agent Manifest 测试报告

## 测试时间

2026-08-21

## 测试范围

本次测试覆盖以下内容：

- Agent manifest 是否能正确加载。
- `iot_agent` 的 `allowed_actions` 是否只包含 `query_device_by_ip`。
- Agent 构建时是否只获得 allowlist 中的工具。
- 未注册 Agent 是否被拒绝。
- Action Gateway 是否允许合法的设备查询请求。
- 非法 IPv4、缺少必填参数和未知参数是否被拒绝。
- 未注册 Action 是否被拒绝。
- Gateway 校验失败时是否不会调用外部 executor。

测试不依赖数据库，也不调用真实 NMS API。HTTP executor 使用 mock。

## 配置职责

- `config/actions.yaml` 是业务 Action 输入 schema、风险、前置条件和效果的权威来源。
- `config/agents.yaml` 是 Agent 身份和 `allowed_actions` 的权威来源。
- `config/tools.yaml` 只保存底层 HTTP executor 的 endpoint、method、参数位置、超时和重试配置，不重复定义业务参数 schema。

## 测试脚本指令

在项目根目录 `/home/user/G-Agent` 执行：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -q
```

辅助语法和差异检查：

```bash
.venv/bin/python -m compileall -q app tests
git diff --check
```

## 测试文件

- `tests/ontology/test_manifest.py`
- `tests/ontology/test_action_gateway.py`

## 测试用例结果

| 测试内容 | 结果 |
|---|---|
| `iot_agent` manifest 能正常加载 | 通过 |
| `iot_agent` 只声明 `query_device_by_ip` | 通过 |
| Agent 工具集合只包含 allowlist 中的 Action | 通过 |
| 未注册 Agent 被拒绝 | 通过 |
| 合法 IPv4 请求可以调用 executor | 通过 |
| 非法 IPv4 请求被拒绝 | 通过 |
| 非法 IPv4 不会调用 executor | 通过 |
| 未注册 Action 被拒绝 | 通过 |
| 未注册 Action 不会调用 executor | 通过 |
| 缺少必填参数被拒绝 | 通过 |
| 包含未知参数被拒绝 | 通过 |
| 参数校验失败时不会调用 executor | 通过 |

其中部分拒绝行为在同一个参数化测试流程中验证，因此 unittest 实际统计为 7 个测试方法。

## 最终结果

```text
Ran 7 tests in 0.023s

OK
```

- 总测试方法：7
- 通过：7
- 失败：0
- 错误：0
- 结论：测试通过

## 验证结论

当前实现已验证以下链路：

```text
Agent manifest
  -> allowed_actions 裁剪工具
  -> Action Gateway 再次校验
  -> 参数和 IPv4 前置条件校验
  -> 校验通过后调用 executor
```

校验失败时不会触发外部 API 请求，未授权或未注册的 Action 不会执行。
