#!/usr/bin/env bash
# Merge the HF part files (part_aa + part_ab) into the final FAISS index.
#
# The official index is split into two parts for download stability; the
# parts are NOT usable alone — merge them first, then point --index_path
# at the merged single file.
#
# Usage: put this script next to your part files (or set WORK_DIR), then:
#   bash merge_index.sh

set -euo pipefail

# ========================== 配置区 ==========================
# 分片所在目录（默认 = 本脚本所在目录）
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 合并后的文件名（e5_Flat 用 e5_Flat.index；HNSW64 分片改成 e5_HNSW64.index）
INDEX_NAME="e5_Flat.index"

# 合并后是否删除分片（省 64G 磁盘；想留备份就设 yes）
KEEP_PARTS="no"
# ============================================================

cd "$WORK_DIR"

if [ ! -f part_aa ] || [ ! -f part_ab ]; then
    echo "错误: 缺少 part_aa 或 part_ab（当前目录: $WORK_DIR）"
    echo "  检查: ls -lh part_aa part_ab"
    exit 1
fi

echo "==> 合并 part_aa + part_ab -> $INDEX_NAME"
cat part_aa part_ab > "$INDEX_NAME"

size=$(ls -lh "$INDEX_NAME" | awk '{print $5}')
echo "==> 完成: $INDEX_NAME ($size)"

if [ "$KEEP_PARTS" = "no" ]; then
    rm part_aa part_ab
    echo "已删除分片"
fi
