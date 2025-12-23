import os
from collections import defaultdict

import pandas as pd

city = "CHI"
raw_UrbanKG_path = f"./raw_data/{city}/UrbanKG_{city}.txt"
entity2id_path = f"./raw_data/{city}/entity2id_{city}.txt"
UrbanKG_path = f"./raw_data/{city}/{city}_assist_kg.csv"

rel_list = ["ALB", "ANA", "BNB", "JBB", "JBR", "JHJC", "JLA", "PBB", "PHPC", "PLA", "RBB", "RHRC", "RLA"]

def save_UrbanKG():
    os.makedirs(os.path.dirname(raw_UrbanKG_path), exist_ok=True)
    kg_lst, category_of_poi, area_list = [], {}, []
    pois_of_area = defaultdict(list)
    with open(entity2id_path, 'r') as f:
        for line in f:
            if line.startswith("Area"):
                area_list.append(line.split()[0].split('/')[-1])
    area_list.sort()
    with open(raw_UrbanKG_path, 'r') as f:
        for line in f:
            for rel in rel_list:
                if rel in line:
                    mid = line.find(rel)
                    head, tail = line[:mid-1], line[mid+len(rel)+1:-1]
                    head_id, tail_id = head.split('/')[-1], tail.split('/')[-1]
                    if rel == "ANA":
                        kg_lst.append([head_id, "adj", tail_id])
                    if rel == "PHPC":
                        category_of_poi[head_id] = tail_id
                    if rel == "PLA":
                       pois_of_area[tail_id].append(head_id)
    area_pois_count = defaultdict(int)
    for area_id, pois in pois_of_area.items():
        for poi in pois:
            area_pois_count[(area_id, category_of_poi[poi])] += 1
    for k, num in area_pois_count.items():
        kg_lst.append([k[0], num, k[1]])
    df = pd.DataFrame(kg_lst, columns=["head", "rel", "tail"])
    df.to_csv(UrbanKG_path, index=False, header=False)

def main():
    save_UrbanKG()

if __name__ == '__main__':
    main()