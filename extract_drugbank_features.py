
#!/usr/bin/env python
# coding: utf-8

import xml.etree.ElementTree as ET
import pandas as pd

# 文件路径
XML_PATH = "./DrugBank_all_data.xml"
DRUG_LIST_CSV = "./deepDDI/drug_list.csv"
OUTPUT_CSV = "./deepDDI/Ryu_drug_features.csv"


def load_drug_list(drug_list_path):
    """从CSV加载药物ID和SMILES"""
    df = pd.read_csv(drug_list_path, dtype=str)

    # 确定列名
    if 'drugbank_id' in df.columns:
        id_col = 'drugbank_id'
    elif 'id' in df.columns:
        id_col = 'id'
    else:
        id_col = df.columns[0]

    if 'smiles' in df.columns:
        smiles_col = 'smiles'
    elif 'SMILES' in df.columns:
        smiles_col = 'SMILES'
    else:
        smiles_col = df.columns[1]

    # 构建药物信息字典
    drug_info = {}

    for _, row in df.iterrows():
        drug_id = str(row[id_col]).strip()

        if not drug_id or drug_id.lower() == 'nan':
            continue

        smiles = (
            str(row[smiles_col]).strip()
            if pd.notna(row[smiles_col]) else ''
        )

        drug_info[drug_id] = {
            'smiles': smiles,
            'target': '',
            'enzyme': '',
            'pathway': ''
        }

    print(f"从CSV加载了 {len(drug_info)} 个药物")
    return drug_info


def parse_drugbank_xml(xml_path, drug_info):
    """从DrugBank XML提取Target、Enzyme、Pathway"""

    ns = {'db': 'http://www.drugbank.ca'}
    drug_tag = f'{{{ns["db"]}}}drug'

    target_set = set(drug_info.keys())

    print(f"需要处理的药物数量: {len(target_set)}")
    print("开始解析XML文件...")

    found_ids = set()

    # 流式解析
    context = ET.iterparse(
        xml_path, events=('start', 'end')
    )
    _, root = next(context)

    for event, elem in context:

        if event == 'end' and elem.tag == drug_tag:

            # 1. 获取DrugBank ID
            drugbank_id = None

            for id_elem in elem.findall('db:drugbank-id', ns):
                if (
                    id_elem.text
                    and id_elem.get('primary') == 'true'
                ):
                    drugbank_id = id_elem.text.strip()
                    break

            if not drugbank_id:
                id_elem = elem.find('db:drugbank-id', ns)
                if id_elem is not None and id_elem.text:
                    drugbank_id = id_elem.text.strip()

            # 只处理目标药物
            if drugbank_id in target_set:

                found_ids.add(drugbank_id)

                # 2. 提取Target
                targets = []

                targets_elem = elem.find('db:targets', ns)

                if targets_elem is not None:
                    for target in targets_elem.findall(
                        'db:target', ns
                    ):
                        polypeptide = target.find(
                            'db:polypeptide', ns
                        )

                        if polypeptide is not None:
                            uniprot_id = polypeptide.get('id')

                            if uniprot_id:
                                targets.append(uniprot_id)

                drug_info[drugbank_id]['target'] = '|'.join(
                    dict.fromkeys(targets)
                )

                # 3. 提取Enzyme
                enzymes = []

                enzymes_elem = elem.find('db:enzymes', ns)

                if enzymes_elem is not None:
                    for enzyme in enzymes_elem.findall(
                        'db:enzyme', ns
                    ):
                        polypeptide = enzyme.find(
                            'db:polypeptide', ns
                        )

                        if polypeptide is not None:
                            uniprot_id = polypeptide.get('id')

                            if uniprot_id:
                                enzymes.append(uniprot_id)

                drug_info[drugbank_id]['enzyme'] = '|'.join(
                    dict.fromkeys(enzymes)
                )

                # 4. 新增：提取Pathway
                pathways = []

                pathways_elem = elem.find(
                    'db:pathways', ns
                )

                if pathways_elem is not None:
                    for pathway in pathways_elem.findall(
                        'db:pathway', ns
                    ):
                        pathway_id = pathway.findtext(
                            'db:smpdb-id',
                            default='',
                            namespaces=ns
                        )

                        if pathway_id:
                            pathways.append(pathway_id.strip())

                drug_info[drugbank_id]['pathway'] = '|'.join(
                    dict.fromkeys(pathways)
                )

            # 清理内存
            root.clear()

    print(f"在XML中找到的药物数量: {len(found_ids)}")
    print(f"未匹配药物数量: {len(target_set - found_ids)}")

    return drug_info


def main():

    # 1. 加载药物列表
    print("加载药物列表...")
    drug_info = load_drug_list(DRUG_LIST_CSV)

    # 2. 从XML提取Target、Enzyme、Pathway
    drug_info = parse_drugbank_xml(
        XML_PATH, drug_info
    )

    # 3. 保存结果
    results = []

    for drug_id, info in drug_info.items():
        results.append({
            'drugbank_id': drug_id,
            'smiles': info['smiles'],
            'target': info['target'],
            'enzyme': info['enzyme'],
            'pathway': info['pathway']
        })

    df_results = pd.DataFrame(results)

    df_results.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding='utf-8-sig'
    )

    # 4. 统计信息
    print("\n========== 统计信息 ==========")

    total = len(df_results)

    print(f"总药物数: {total}")

    for col in ['smiles', 'target', 'enzyme', 'pathway']:
        count = (df_results[col] != '').sum()
        ratio = count / total * 100 if total else 0

        print(
            f"有{col}: {count}/{total} "
            f"({ratio:.2f}%)"
        )

    # 同时具有Target、Enzyme、Pathway
    complete = (
        (df_results['target'] != '') &
        (df_results['enzyme'] != '') &
        (df_results['pathway'] != '')
    ).sum()

    print(f"同时具有Target、Enzyme、Pathway: {complete}")

    # 统计Pathway唯一ID数量
    unique_pathways = set()
    total_pathway_links = 0

    for value in df_results['pathway']:
        if value:
            ids = value.split('|')
            unique_pathways.update(ids)
            total_pathway_links += len(ids)

    print(f"不同Pathway ID数量: {len(unique_pathways)}")
    print(f"药物-Pathway关联总数: {total_pathway_links}")

    print(f"\n结果已保存至: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
