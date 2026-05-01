#!/usr/bin/env python3
"""
Script to convert wgsl-rs macro-based WGSL to raw WGSL strings.
"""

import re
import os

def convert_file(filepath):
    """Convert a single rival file from wgsl-rs macros to raw WGSL."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Remove top-level wgsl_rs import
    content = re.sub(r'use wgsl_rs::wgsl;\n', '', content)
    content = re.sub(r'use wgsl_rs::wgsl;', '', content)
    
    # Pattern to match #[wgsl] pub mod X { ... }
    # This is complex - we need to find the module and convert it
    
    # First, find all #[wgsl] modules
    wgsl_modules = []
    
    # Pattern for the module declaration
    module_pattern = r'#\[wgsl\]\s*pub mod (\w+)\s*\{(.*?)\n\}'
    
    # This is getting complex. Let me try a different approach - 
    # just print what needs to be done for each file
    
    print(f"File: {filepath}")
    
    # Count wgsl_rs references
    wgsl_count = content.count('wgsl_rs')
    wgsl_macro_count = content.count('#[wgsl]')
    
    print(f"  wgsl_rs references: {wgsl_count}")
    print(f"  #[wgsl] macros: {wgsl_macro_count}")
    
    # Find lines with wgsl_rs
    for i, line in enumerate(content.split('\n'), 1):
        if 'wgsl_rs' in line:
            print(f"  Line {i}: {line.strip()}")
    
    print()

# Convert all rival files
rival_dir = 'src/rivals'
for filename in os.listdir(rival_dir):
    if filename.endswith('.rs'):
        filepath = os.path.join(rival_dir, filename)
        convert_file(filepath)
