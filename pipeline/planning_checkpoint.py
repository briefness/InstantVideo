"""规划步骤的持久化回放；不重试失败，也不把规划记录当作验收结论。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
import stat

import config
from pipeline.structured_response import load_unique_json


FILENAME = 'planning_checkpoint.json'
VERSION = 'planning-checkpoint-v1'
MAX_BYTES = 32 * 1024 * 1024
MAX_STEPS = 512


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


class PlanningCheckpoint:
    """由现有工作区锁保护顺序步骤，写前记录请求意图，写后记录完整结果。"""

    def __init__(self, workspace, *, resume: bool):
        self.workspace = workspace
        self.path = workspace.path / FILENAME
        self.cursor = 0
        self.context = _digest({'version': VERSION, 'options': workspace.options.model_dump(mode='json'),
                                'model': config.LLM_MODEL, 'endpoint': config.ARK_BASE_URL})
        with workspace._thread_lock, workspace._file_lock('.planning.lock', nonblocking=True):
            if resume:
                self._read()
            elif self.path.exists() or self.path.is_symlink():
                raise ValueError('规划记录已存在；必须使用正式恢复入口，不能覆盖已有成果')
            else:
                self._write({'version': VERSION, 'context': self.context, 'steps': []})

    def _write(self, record: dict) -> None:
        record = {k: v for k, v in record.items() if k != 'digest'}
        record['digest'] = _digest(record)
        if len(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).encode()) > MAX_BYTES:
            raise ValueError('规划记录超出本地容量上限；未重复调用模型')
        self.workspace._atomic_write_json(self.path, record)

    def _read(self) -> dict:
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as handle:
                info = os.fstat(handle.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
                    raise ValueError('invalid checkpoint file')
                data = handle.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError('oversized checkpoint')
            record = load_unique_json(data.decode('utf-8'))
            if (not isinstance(record, dict) or set(record) != {'version', 'context', 'steps', 'digest'}
                    or record['version'] != VERSION or record['context'] != self.context
                    or record['digest'] != _digest({k: v for k, v in record.items() if k != 'digest'})
                    or not isinstance(record['steps'], list) or len(record['steps']) > MAX_STEPS):
                raise ValueError('invalid checkpoint context')
            names = set()
            for item in record['steps']:
                if not isinstance(item, dict):
                    raise ValueError('invalid checkpoint step')
                state = item.get('state')
                fields = {'key', 'input_hash', 'state'}
                if state == 'returned':
                    fields |= {'value', 'value_hash'}
                    if not isinstance(item.get('value'), dict) or _digest(item['value']) != item.get('value_hash'):
                        raise ValueError('changed checkpoint value')
                elif state == 'failed':
                    fields.add('error_type')
                    if not isinstance(item.get('error_type'), str):
                        raise ValueError('invalid failure')
                elif state != 'started':
                    raise ValueError('unknown checkpoint state')
                if (set(item) != fields or not isinstance(item['key'], str) or not item['key']
                        or len(item['key']) > 128 or item['key'] in names
                        or not isinstance(item['input_hash'], str) or len(item['input_hash']) != 64):
                    raise ValueError('invalid checkpoint identity')
                names.add(item['key'])
            return record
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError('规划进度缺失、损坏或与当前需求/模型配置不一致；未重新调用模型') from exc

    def step(self, key: str, inputs: dict, operation) -> dict:
        signature = _digest(inputs)
        with self.workspace._thread_lock, self.workspace._file_lock('.planning.lock', nonblocking=True):
            record = self._read()
            steps = record['steps']
            if self.cursor < len(steps):
                entry = steps[self.cursor]
                if entry['key'] != key or entry['input_hash'] != signature:
                    raise ValueError('规划步骤的输入或顺序已变化；不能复用旧结果或自动重新生成')
                if entry['state'] != 'returned':
                    raise ValueError('规划步骤已失败或请求结果不明；保留现场，未自动重发模型请求')
                self.cursor += 1
                return deepcopy(entry['value'])
            if self.cursor != len(steps) or len(steps) >= MAX_STEPS or any(e['key'] == key for e in steps):
                raise ValueError('规划步骤重复或超出上限；未调用模型')
            entry = {'key': key, 'input_hash': signature, 'state': 'started'}
            steps.append(entry)
            self._write(record)  # 必须先落盘；崩溃后的 started 不能当成未提交。
            try:
                value = operation()
                if not isinstance(value, dict):
                    raise ValueError('规划步骤必须返回完整对象')
                digest = _digest(value)
            except Exception as exc:
                entry.update(state='failed', error_type=type(exc).__name__)
                self._write(record)
                raise
            entry.update(state='returned', value=deepcopy(value), value_hash=digest)
            self._write(record)
            self.cursor += 1
            return deepcopy(value)
