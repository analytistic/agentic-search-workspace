"""极简 .env 读取：不引入 python-dotenv，只处理 KEY=VALUE 这种最常见格式。

优先级：已有的环境变量 > .env > 代码里的默认值。
这样命令行 export 的值永远压过文件，方便临时覆盖。
"""

from __future__ import annotations

import os
from typing import Optional


def find_env_file(start: Optional[str] = None) -> Optional[str]:
    """从当前目录逐级向上找 .env，找不到返回 None。"""
    cur = os.path.abspath(start or os.getcwd())
    while True:
        cand = os.path.join(cur, ".env")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def load_env(path: Optional[str] = None) -> dict[str, str]:
    """读 .env 并写进 os.environ，返回本次真正写入的键值。"""
    path = path or find_env_file()
    if not path or not os.path.isfile(path):
        return {}

    loaded: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key or key in os.environ or value == "":
                continue
            os.environ[key] = value
            loaded[key] = value
    return loaded


def env(name: str, default=None):
    """.env 里的键取出来，没有就用默认值。"""
    value = os.environ.get(name)
    return value if value not in (None, "") else default
