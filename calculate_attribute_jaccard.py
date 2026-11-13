#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 drug_list.csv 的行顺序生成三种属性的 Jaccard 相似度矩阵。

直接运行本脚本即可；默认输入、输出路径均相对于脚本所在目录。
输出与 chem_Jacarrd_sim.csv 一致：纯数字、无表头、无行索引。
所有药物自身相似度（对角线）统一为 1，包括属性缺失的药物。不同药物之间，任一属性集合为空时相似度为 0。
使用 --check-only 仅检查输入，不计算或保存矩阵。
默认拒绝覆盖已有输出；确认需要替换时使用 --overwrite。
"""
import argparse
import csv
from pathlib import Path

import numpy as np

ATTRIBUTES = ('enzyme', 'pathway', 'target')


def read_records(path, required):
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: 缺少列 {sorted(missing)}')
        records = list(reader)
    if not records:
        raise ValueError(f'{path}: 文件没有数据')
    seen = set()
    for row_number, record in enumerate(records, 2):
        drug_id = (record.get('drugbank_id') or '').strip()
        if not drug_id:
            raise ValueError(f'{path}: 第 {row_number} 行缺少 drugbank_id')
        if drug_id in seen:
            raise ValueError(f'{path}: 重复药物编号 {drug_id}')
        seen.add(drug_id)
        record['drugbank_id'] = drug_id
    return records


def parse_entities(value):
    # CSV 空白单元格代表缺失；不把缺失建立为一个共享的实体。
    missing = {'', 'nan', 'none', 'null', 'n/a', 'na'}
    return frozenset(
        entity.strip() for entity in (value or '').split('|')
        if entity.strip().lower() not in missing
    )


def jaccard_matrix(entity_sets):
    """非对角线按集合交并比计算；对角线约定为 1，缺失属性不建立跨药物相似关系。"""
    size = len(entity_sets)
    matrix = np.eye(size, dtype=np.float64)
    for i, left in enumerate(entity_sets):
        if not left:
            continue
        for j in range(i + 1, size):
            right = entity_sets[j]
            if not right:
                continue
            intersection = len(left & right)
            union = len(left) + len(right) - intersection
            matrix[i, j] = matrix[j, i] = intersection / union
    return matrix


def check_reference(path, size):
    # 流式检查，不把较大的参考矩阵整体载入内存。
    rows = 0
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        for rows, row in enumerate(csv.reader(stream), 1):
            if len(row) != size:
                raise ValueError(f'{path}: 第 {rows} 行有 {len(row)} 列，应为 {size}')
            try:
                values = np.asarray(row, dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f'{path}: 应为无表头、无索引的纯数字矩阵') from exc
            if not np.isfinite(values).all():
                raise ValueError(f'{path}: 第 {rows} 行含非有限值')
    if rows != size:
        raise ValueError(f'{path}: 有 {rows} 行，应为 {size}')


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=base / 'deepDDI/Ryu_drug_features.csv')
    parser.add_argument('--drug-list', type=Path, default=base / 'deepDDI/drug_list.csv')
    parser.add_argument('--reference', type=Path, default=base / 'deepDDI/chem_Jacarrd_sim.csv')
    parser.add_argument('--output-dir', type=Path, default=base)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    drugs = read_records(args.drug_list, ('drugbank_id',))
    features = read_records(args.features, ('drugbank_id',) + ATTRIBUTES)
    drug_ids = [record['drugbank_id'] for record in drugs]
    by_id = {record['drugbank_id']: record for record in features}
    absent = set(drug_ids) - set(by_id)
    if absent:
        raise ValueError(f'属性文件缺少 {len(absent)} 个药物，例如 {sorted(absent)[:5]}')
    extra = set(by_id) - set(drug_ids)
    if extra:
        print(f'提示：忽略属性文件中不在药物列表内的 {len(extra)} 个药物。')
    aligned = [by_id[drug_id] for drug_id in drug_ids]
    check_reference(args.reference, len(drug_ids))
    print(f'已按 drug_list.csv 顺序对齐 {len(drug_ids)} 个药物。')
    print(f'参考格式检查通过：{len(drug_ids)} x {len(drug_ids)}，无表头、无索引。')
    if args.features.resolve() == args.drug_list.resolve():
        raise ValueError('属性文件与药物列表不能是同一文件')
    inputs = {path.resolve() for path in (args.features, args.drug_list, args.reference)}
    outputs = {attr: args.output_dir / f'{attr}_Jacarrd_sim.csv' for attr in ATTRIBUTES}
    for path in outputs.values():
        if path.resolve() in inputs:
            raise ValueError(f'输出路径与输入冲突：{path}')
        if not args.check_only and path.exists() and not args.overwrite:
            raise FileExistsError(f'输出已存在：{path}；需要覆盖时添加 --overwrite')
    if not args.output_dir.is_dir():
        raise NotADirectoryError(f'输出目录不存在：{args.output_dir}')

    for attr in ATTRIBUTES:
        entity_sets = [parse_entities(record[attr]) for record in aligned]
        populated = sum(bool(entities) for entities in entity_sets)
        vocabulary = set().union(*entity_sets)
        print(f'{attr}: 非空 {populated}/{len(drug_ids)} '
              f'({populated / len(drug_ids):.2%})，不同实体 {len(vocabulary)} 个')
        if args.check_only:
            continue
        matrix = jaccard_matrix(entity_sets)
        if not np.isfinite(matrix).all() or not np.array_equal(matrix, matrix.T):
            raise RuntimeError(f'{attr}: 矩阵校验失败')
        if matrix.min() < 0.0 or matrix.max() > 1.0:
            raise RuntimeError(f'{attr}: 相似度超出 [0, 1]')
        if not np.all(np.diag(matrix) == 1.0):
            raise RuntimeError(f'{attr}: 对角线应全部为 1')
        # 科学计数法保留16位小数，与现有化学相似度文件的文本格式一致。
        np.savetxt(outputs[attr], matrix, delimiter=',', fmt='%.16e', encoding='utf-8')
        print(f'已保存：{outputs[attr]}，形状 {matrix.shape}')
    if args.check_only:
        print('仅检查完成，未生成相似度文件。')


if __name__ == '__main__':
    main()