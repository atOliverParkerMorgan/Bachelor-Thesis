import os
import argparse
import xml.etree.ElementTree as ET

# ==========================================
# Strict Hierarchy Mapping
# ==========================================
Z_ORDER_MAP = {
    "Zdravé dřevo": "0",
    "Pozadí": "1",
    "Suk": "2",
    "Hniloba": "3",
    "Kůra": "4",
    "Trhlina": "5",
    "Poškození hmyzem": "6",
}


def _append_full_frame_zdrave_drevo(image: ET.Element) -> bool:
    """Add a healthy-wood polygon that covers the full frame."""
    width = image.get("width")
    height = image.get("height")
    if width is None or height is None:
        return False

    try:
        max_x = max(0, int(width) - 1)
        max_y = max(0, int(height) - 1)
    except ValueError:
        return False

    polygon = ET.SubElement(image, "polygon")
    polygon.set("label", "Zdravé dřevo")
    polygon.set("occluded", "0")
    polygon.set("source", "manual")
    polygon.set("z_order", Z_ORDER_MAP["Zdravé dřevo"])
    polygon.set("points", f"0,0;{max_x},0;{max_x},{max_y};0,{max_y}")
    return True


def _is_zdrave_label(label: str | None) -> bool:
    if label is None:
        return False
    normalized = label.strip().lower()
    return normalized in {"zdrave_drevo", "zdrave drevo", "Zdravé dřevo", "zdravé dřevo"}


def _remove_existing_zdrave_annotations(image: ET.Element) -> int:
    removed = 0
    for shape in list(image):
        if _is_zdrave_label(shape.get("label")):
            image.remove(shape)
            removed += 1
    return removed

def fix_cvat_z_order_and_ungroup(tree_name):
    tree_lower = tree_name.lower()
    
    # Define dynamic paths
    xml_file = f"src/cvat_exports/cvat/{tree_lower}/annotations.xml"
    output_file = f"src/cvat_exports/cvat/{tree_lower}/annotations_fixed.xml"

    if not os.path.exists(xml_file):
        alt_xml_file = f"src/cvat_exports/cvat/{tree_lower}_seg/annotations.xml"
        if os.path.exists(alt_xml_file):
            xml_file = alt_xml_file
            output_file = f"src/cvat_exports/cvat/{tree_lower}_seg/annotations_fixed.xml"
        else:
            print(f"Error: Could not find XML file at {xml_file}")
            return

    print(f"Loading {xml_file}...")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    z_order_changes = 0
    ungrouped_count = 0
    zdrave_added_count = 0
    zdrave_removed_count = 0

    # Look through every frame (image)
    for image in root.findall('image'):
        zdrave_removed_count += _remove_existing_zdrave_annotations(image)
        if _append_full_frame_zdrave_drevo(image):
            zdrave_added_count += 1

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
    print(f"Removed {zdrave_removed_count} existing Zdravé dřevo shapes before regeneration.")
    print(f"Added full-frame Zdravé dřevo polygons on {zdrave_added_count} images.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix CVAT XML z-order and ungroup objects.")
    parser.add_argument("tree", type=str, help="The identifier of the tree (e.g., dub_1, dub_5)")
    args = parser.parse_args()
    
    fix_cvat_z_order_and_ungroup(args.tree)