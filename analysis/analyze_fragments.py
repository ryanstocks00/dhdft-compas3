#!/usr/bin/env python3
"""
Analyze fragment statistics for JSON files in fionas_systems folder.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def get_fragments_from_json(filepath):
    """Extract fragments from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if fragments are directly in the root
    if 'fragments' in data:
        return data['fragments']
    
    # Check if fragments are in topologies
    if 'topologies' in data and len(data['topologies']) > 0:
        if 'fragments' in data['topologies'][0]:
            return data['topologies'][0]['fragments']
    
    return None

def analyze_fragments(filepath):
    """Analyze fragments in a JSON file."""
    fragments = get_fragments_from_json(filepath)
    
    if fragments is None:
        return None
    
    num_fragments = len(fragments)
    atom_counts = [len(frag) for frag in fragments]
    
    if num_fragments == 0:
        return {
            'num_fragments': 0,
            'min_atoms': None,
            'avg_atoms': None,
            'max_atoms': None
        }
    
    return {
        'num_fragments': num_fragments,
        'min_atoms': min(atom_counts),
        'avg_atoms': sum(atom_counts) / num_fragments,
        'max_atoms': max(atom_counts)
    }

def main():
    # Find all JSON files in fionas_systems
    base_dir = Path('/mnt/c/Users/ryans/code/dhdft/inputs/fionas_systems')
    json_files = sorted(base_dir.rglob('*.json'))
    
    results = []
    
    for json_file in json_files:
        rel_path = json_file.relative_to(base_dir)
        stats = analyze_fragments(json_file)
        
        if stats is None:
            print(f"Warning: Could not extract fragments from {rel_path}")
            continue
        
        results.append({
            'file': str(rel_path),
            'stats': stats
        })
    
    # Print results
    print(f"{'File':<50} {'Fragments':<12} {'Min Atoms':<12} {'Avg Atoms':<12} {'Max Atoms':<12}")
    print("=" * 100)
    
    for result in results:
        stats = result['stats']
        if stats['num_fragments'] == 0:
            print(f"{result['file']:<50} {stats['num_fragments']:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        else:
            print(f"{result['file']:<50} {stats['num_fragments']:<12} {stats['min_atoms']:<12} {stats['avg_atoms']:<12.2f} {stats['max_atoms']:<12}")
    
    return results

if __name__ == '__main__':
    main()

