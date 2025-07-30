import os
import argparse
from thefuzz import fuzz

def parse_alignments(filepath):
    """Parse print1/print2 format alignment files, returning [(TKG node string, KG node string)]"""
    alignments = []
    tkg = kg = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('TKG Node:'): # Changed from 'TKG节点:'
                tkg = line
            elif line.startswith('KG Node:'): # Changed from 'KG节点:'
                kg = line
            elif line == 'ALIGN_TO' or not line:
                continue
            if tkg and kg:
                alignments.append((tkg, kg))
                tkg = kg = None
    return alignments

def extract_id_and_name(node_str):
    # Extract id and name
    id_val = None
    name_val = None
    if 'id=' in node_str:
        s = node_str.split('id=')[1]
        id_val = s.split(',')[0].strip()
    if 'name=' in node_str:
        s = node_str.split('name=')[1]
        name_val = s.split(',')[0].strip()
    return id_val, name_val

def find_best_match(node_str, print1_nodes):
    """
    Find the most similar node string in print1_nodes.
    Matching priority: exact id match > exact name match > highest name similarity.
    """
    id2, name2 = extract_id_and_name(node_str)
    # 1. Exact id match
    for n in print1_nodes:
        id1, name1 = extract_id_and_name(n)
        if id1 and id2 and id1 == id2:
            return n
    # 2. Exact name match
    for n in print1_nodes:
        id1, name1 = extract_id_and_name(n)
        if name1 and name2 and name1 == name2:
            return n
    # 3. Highest name similarity
    best_n = None
    best_score = -1
    for n in print1_nodes:
        id1, name1 = extract_id_and_name(n)
        if name1 and name2:
            score = fuzz.token_set_ratio(name1, name2)
            if score > best_score:
                best_score = score
                best_n = n
    if best_score >= 80:
        return best_n
    # 4. fallback: return the most similar
    return best_n if best_n else node_str

def fix_print2_by_print1(print1_dir, print2_dir, output_dir):
    """
    Fixes the node information in print2 files by finding the best match from corresponding print1 files.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(print2_dir) if f.endswith('.txt')]
    for fname in files:
        print(f"Fixing: {fname}")
        print1_path = os.path.join(print1_dir, fname)
        print2_path = os.path.join(print2_dir, fname)
        output_path = os.path.join(output_dir, fname)
        if not os.path.exists(print1_path) or not os.path.exists(print2_path):
            print(f"   Skipping: {fname} (print1 or print2 file does not exist)")
            continue
        print1_aligns = parse_alignments(print1_path)
        print2_aligns = parse_alignments(print2_path)
        print1_tkg_nodes = [a[0] for a in print1_aligns]
        print1_kg_nodes = [a[1] for a in print1_aligns]
        with open(output_path, 'w', encoding='utf-8') as f:
            for tkg2, kg2 in print2_aligns:
                best_tkg = find_best_match(tkg2, print1_tkg_nodes)
                best_kg = find_best_match(kg2, print1_kg_nodes)
                f.write(f"{best_tkg}\nALIGN_TO\n{best_kg}\n\n")
        print(f"   Fixed and saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix node information in the print2 folder using the most similar match from print1')
    parser.add_argument('--print1_dir', type=str, default='print1', help='Path to the print1 folder')
    parser.add_argument('--print2_dir', type=str, default='print3', help='Path to the folder to be fixed (e.g., print2 or print3)')
    parser.add_argument('--output_dir', type=str, default='print3_repair', help='Output folder path for the fixed version (default: print3_repair)')
    args = parser.parse_args()
    fix_print2_by_print1(args.print1_dir, args.print2_dir, args.output_dir)