import os
import argparse
import xml.etree.ElementTree as ET

# ==========================================
# Strict Hierarchy Mapping
# ==========================================
Z_ORDER_MAP = {
    "Pozadi": "0",
    "Kura": "1",
    "Poškození hmyzem": "2", 
    "Trhlina": "3",
    "Suk": "4",
    "Hniloba": "5"
}

def fix_cvat_z_order_and_ungroup(tree_name):
    tree_lower = tree_name.lower()
    
    # Define dynamic paths
    xml_file = f"src/cvat_exports/{tree_lower}/annotations.xml"
    output_file = f"src/cvat_exports/{tree_lower}/annotations_fixed.xml"

    if not os.path.exists(xml_file):
        alt_xml_file = f"src/cvat_exports/{tree_lower}_seg/annotations.xml"
        if os.path.exists(alt_xml_file):
            xml_file = alt_xml_file
            output_file = f"src/cvat_exports/{tree_lower}_seg/annotations_fixed.xml"
        else:
            print(f"Error: Could not find XML file at {xml_file}")
            return

    print(f"Loading {xml_file}...")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    z_order_changes = 0
    ungrouped_count = 0

    # Look through every frame (image)
    for image in root.findall('image'):
        # Look through every shape (mask, polygon, etc.)
        for shape in image:
            # 1. Fix the Z-Order
            label = shape.get('label')
            if label in Z_ORDER_MAP:
                shape.set('z_order', Z_ORDER_MAP[label])
                z_order_changes += 1
            
            # 2. DELETE the group_id so CVAT stops merging them!
            if 'group_id' in shape.attrib:
                del shape.attrib['group_id']
                ungrouped_count += 1

    # Save the new file
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Success! Updated z_order for {z_order_changes} objects.")
    print(f"Removed group_id from {ungrouped_count} objects to prevent CVAT export bugs.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix CVAT XML z-order and ungroup objects.")
    parser.add_argument("tree", type=str, help="The identifier of the tree (e.g., dub_1, dub_5)")
    args = parser.parse_args()
    
    fix_cvat_z_order_and_ungroup(args.tree)