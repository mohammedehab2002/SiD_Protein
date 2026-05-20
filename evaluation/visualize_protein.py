import os
from pymol import cmd

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = "/homes/kasram/broteina/SiD_Protein/eval_tmp/network-snapshot-1.000000-001892.pkl/250/"
OUTPUT_FOLDER = "textbook_renders"   
RESOLUTION = (1200, 1200)   

def apply_textbook_style():
    # 1. Calculate Secondary Structure (Critical for arrows/helices)
    cmd.dss()
    cmd.hide("everything")
    cmd.show("cartoon")
    cmd.set("cartoon_ca_only", 1)

    # ==========================================
    # A. GEOMETRY (Thin Loops, Wide Helices)
    # ==========================================
    # Make the helices (oval) and sheets (rect) wide and flat
    cmd.set("cartoon_oval_length", 1.50)  # Very wide helices
    cmd.set("cartoon_oval_width",  0.25)  # Thin/flat profile
    cmd.set("cartoon_rect_length", 1.50)  # Wide sheets
    cmd.set("cartoon_rect_width",  0.25)
    
    # Make the loops (coils) very thin "wires"
    cmd.set("cartoon_loop_radius", 0.12)  # <--- This creates the contrast you asked for

    # Enable "fancy" helices to see the inside/outside distinction
    cmd.set("cartoon_fancy_helices", 1) 
    cmd.set("cartoon_dumbbell_length", 1.5) # Emphasize the dumbell shape

    # ==========================================
    # B. COLORING (Red/Blue/Green + Two-Tone)
    # ==========================================
    # 1. Set the "Inside" color (The grey interior of the helix)
    cmd.set("cartoon_highlight_color", "grey90") 

    # 2. Color by Secondary Structure (SS)
    # 'ss h' = Helix, 'ss s' = Sheet (Arrow), 'ss l+' = Loop
    cmd.color("firebrick", "ss h")   # Dark Red Helices (Outside)
    cmd.color("slate",     "ss s")   # Deep Blue Sheets/Arrows
    cmd.color("forest",    "ss l+")  # Dark Green Loops

    # ==========================================
    # C. LIGHTING & EDGES
    # ==========================================
    # Use standard ray tracing to get clean shadows (not matte/clay)
    cmd.set("ray_trace_mode", 1)     # Mode 1 = Normal (Standard textbook look)
    cmd.set("ray_shadows", 1)        # Shadows on
    cmd.set("light_count", 2)
    cmd.set("spec_reflect", 0.2)     # Slight shine (unlike the matte look)
    
    # Enable black outlines for that "drawn" look (Optional - matching the image's sharpness)
    cmd.set("ray_trace_gain", 0.4)   # Strengthens outlines slightly
    
    # Background
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)

# ==========================================
# MAIN LOOP
# ==========================================
def process_pdbs():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    pdb_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".pdb")]
    
    if not pdb_files:
        print(f"No .pdb files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(pdb_files)} PDB files...")

    for pdb in pdb_files:
        cmd.reinitialize()
        cmd.load(os.path.join(INPUT_FOLDER, pdb), "protein")
        
        apply_textbook_style()
        
        cmd.orient()
        cmd.zoom(complete=1)
        cmd.clip("slab", 100) # Safety clip

        output_name = f"{os.path.splitext(pdb)[0]}.png"
        print(f"Rendering {output_name}...")
        
        cmd.ray(RESOLUTION[0], RESOLUTION[1])
        cmd.png(os.path.join(OUTPUT_FOLDER, output_name))

    print("Done!")

process_pdbs()