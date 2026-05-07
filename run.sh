#!/usr/bin/env bash
set -euo pipefail

# Pipeline: Extract zips → Convert IMA to PNG → Segment → Datumaro → Upload to CVAT

# Default parameters
MASKS="all"
SKIP_EXTRACT=false
SKIP_CONVERT=false
SKIP_SEG=false
SKIP_DATUMARO=false
UPLOAD=false
UPLOAD_JOB_ID=""
TREE=""
DATUMARO_NO_MEDIA=false
CLEAN_LOGS_ONLY=false
series_list=()

# Parse arguments

while [[ $# -gt 0 ]]; do
    case $1 in
        --masks|-m)
            MASKS="$2"
            shift 2
            ;;
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-seg)
            SKIP_SEG=true
            shift
            ;;
        --skip-datumaro)
            SKIP_DATUMARO=true
            shift
            ;;
        --upload)
            UPLOAD=true
            shift
            ;;
        --upload-job-id)
            UPLOAD_JOB_ID="$2"
            shift 2
            ;;
        --tree|-t)
            TREE="$2"
            shift 2
            ;;
        --datumaro-no-media)
            DATUMARO_NO_MEDIA=true
            shift
            ;;
        --clean-logs-only)
            CLEAN_LOGS_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --masks, -m MASKS       Masks to generate (comma-separated internal keys: pozadi,kura,suk,hniloba,trhlina or 'all'; Czech labels are also accepted by segmentation)"
            echo "  --tree, -t TREE         Process only one tree (for example: dub5)"
            echo "  --skip-extract          Skip extraction step if PNG files already exist"
            echo "  --skip-convert          Skip IMA to PNG conversion step"
            echo "  --skip-seg              Skip segmentation if expected masks already exist"
            echo "  --skip-datumaro         Skip Datumaro conversion if output zip already exists"
            echo "  --upload                Upload results to CVAT (requires CVAT_TOKEN and CVAT_PROJECT_ID)"
            echo "  --upload-job-id JOB_ID  Import annotations into existing CVAT job (annotations only)"
            echo "  --datumaro-no-media     Export Datumaro zip without image media (annotations only)"
            echo "  --clean-logs-only       Only remove/report invalid logs (no segmentation)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ -f .env ]; then
    set -a
    source .env 2>/dev/null || true
    set +a
fi

if [ -n "$UPLOAD_JOB_ID" ] || [ -n "${CVAT_JOB_ID:-}" ]; then
    DATUMARO_NO_MEDIA=true
fi

input_root="src/ground_truth"
output_root="src/png"
segmentation_base="src/output"
temp_extract="src/.temp_extract"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Convert masks argument to array format for Python
if [ "$MASKS" = "all" ]; then
    MASK_ARGS="--masks all"
    expected_masks=(pozadi kura suk hniloba trhlina)
else
    MASK_ARGS="--masks ${MASKS//,/ }"
    IFS=',' read -r -a expected_masks <<< "$MASKS"
fi


echo -e "${BLUE}Configuration:${NC}"
echo -e "  Masks to generate: ${MASKS}"
echo -e "  Tree filter: ${TREE:-<all>}"
echo -e "  Skip extract: ${SKIP_EXTRACT}"
echo -e "  Skip convert: ${SKIP_CONVERT}"
echo -e "  Skip segmentation: ${SKIP_SEG}"
echo -e "  Skip Datumaro: ${SKIP_DATUMARO}"
echo -e "  Upload to CVAT: ${UPLOAD}"
echo -e "  Upload Job ID: ${UPLOAD_JOB_ID:-<none>}"
echo -e "  Datumaro no media: ${DATUMARO_NO_MEDIA}"
echo -e "  Clean logs only: ${CLEAN_LOGS_ONLY}"
echo ""

# If only cleaning logs, run segmentation script with --clean-logs-only and exit
if [ "$CLEAN_LOGS_ONLY" = true ]; then
    if [ -z "$TREE" ]; then
        echo -e "${RED}✗ --clean-logs-only requires --tree <tree_name>${NC}"
        exit 1
    fi
    echo -e "${YELLOW}[CLEAN] Removing invalid logs for tree: $TREE${NC}"
    poetry run python -m src.preprocessing.segmentation.segmentation --tree "$TREE" --clean-logs-only
    exit $?
fi

# Cleanup on error
cleanup() {
    if [ -d "$temp_extract" ]; then
        rm -rf "$temp_extract"
    fi
}
trap cleanup EXIT

echo -e "${YELLOW}[1/5] Extracting zip files...${NC}"

# Check if we should skip extraction
if [ "$SKIP_EXTRACT" = true ]; then
    echo -e "${BLUE}  Skipping extraction (--skip-extract enabled)${NC}"

    series_list=()
    if [ -n "$TREE" ]; then
        candidate_dir="$output_root/$TREE"
        if [ -d "$candidate_dir" ]; then
            series_list+=("$TREE")
        fi
    elif [ -d "$output_root" ]; then
        for series_dir in "$output_root"/dub*; do
            if [ -d "$series_dir" ]; then
                series_name=$(basename "$series_dir")
                series_list+=("$series_name")
            fi
        done
    fi

    if [ -n "$TREE" ] && [ ${#series_list[@]} -eq 0 ]; then
        echo -e "${RED}✗ Requested tree '$TREE' not found under $output_root.${NC}"
        exit 1
    fi

    if [ ${#series_list[@]} -eq 0 ]; then
        echo -e "${RED}✗ No existing PNG directories found. Cannot skip extraction.${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Found ${#series_list[@]} existing series${NC}"
else

# Find all zip files
zip_files=($(find "$input_root" -maxdepth 2 -name "*.zip" -type f 2>/dev/null || echo))

if [ ${#zip_files[@]} -eq 0 ]; then
    echo -e "${RED}✗ No zip files found${NC}"
    exit 1
fi

echo "  Found ${#zip_files[@]} zip(s)"

if [ -d "$temp_extract" ]; then
    rm -rf "$temp_extract"
fi
mkdir -p "$temp_extract"

for zip_file in "${zip_files[@]}"; do
    basename="${zip_file##*/}"
    number="${basename//[^0-9]/}"
    series_name="dub$number"

    if [ -n "$TREE" ] && [ "$series_name" != "$TREE" ]; then
        continue
    fi

    extract_dir="$temp_extract/$series_name"

    mkdir -p "$extract_dir"
    unzip -q "$zip_file" -d "$extract_dir"

    ima_files=($(find "$extract_dir" -name "*.IMA" -type f 2>/dev/null || echo))

    if [ ${#ima_files[@]} -gt 0 ]; then
        flat_dir="$temp_extract/${series_name}_flat"
        mkdir -p "$flat_dir"
        for ima_file in "${ima_files[@]}"; do
            cp "$ima_file" "$flat_dir/"
        done
        series_list+=("$series_name")
    fi
done

if [ -n "$TREE" ] && [ ${#series_list[@]} -eq 0 ]; then
    echo -e "${RED}✗ No IMA files found for requested tree '$TREE'.${NC}"
    exit 1
fi

if [ ${#series_list[@]} -eq 0 ]; then
    echo -e "${RED}✗ No IMA files found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Extracted ${#series_list[@]} series${NC}"
fi

echo -e "${YELLOW}[2/5] Converting IMA to PNG...${NC}"

datumaro_files=()

if [ "$SKIP_CONVERT" = true ]; then
    echo -e "${BLUE}  Skipping conversion (--skip-convert enabled)${NC}"
    
    # Check if PNG directories already exist
    for series in "${series_list[@]}"; do
        if [ ! -d "$output_root/$series" ]; then
            echo -e "${YELLOW}  Warning: $series PNG directory not found${NC}"
        fi
    done
    echo -e "${GREEN}✓ Using existing PNG files${NC}"
else
    for series in "${series_list[@]}"; do
        # Check if PNG files already exist
        if [ -d "$output_root/$series" ] && [ "$(find "$output_root/$series" -name '*.png' | wc -l)" -gt 0 ]; then
            echo -e "${BLUE}  $series: PNG files already exist, skipping conversion${NC}"
            continue
        fi
        
        flat_dir="$temp_extract/${series}_flat"
        temp_ground_truth="$temp_extract/ground_truth_$series"
        mkdir -p "$temp_ground_truth/$series"
        cp "$flat_dir"/*.IMA "$temp_ground_truth/$series/" 2>/dev/null || true
        
        if [ -d "$output_root/$series" ]; then
            rm -rf "$output_root/$series"
        fi
        mkdir -p "$output_root"
        
        poetry run python src/preprocessing/conversion/ima2png.py \
            --input "$temp_ground_truth" \
            --output "$output_root" \
            --target "$temp_ground_truth/$series" 2>/dev/null || {
            echo -e "${RED}✗ $series: IMA→PNG failed${NC}"
            continue
        }
    done
    echo -e "${GREEN}✓ PNG conversion complete${NC}"
fi

echo -e "${YELLOW}[3/5] Running segmentation...${NC}"
for series in "${series_list[@]}"; do
    if [ ! -d "$output_root/$series" ]; then
        continue
    fi
    
    segmentation_output="$segmentation_base/$series"

    if [ "$SKIP_SEG" = true ]; then
        existing_masks_ok=true
        for mask_name in "${expected_masks[@]}"; do
            mask_dir="$segmentation_output/masks/$mask_name"
            if [ ! -d "$mask_dir" ] || [ "$(find "$mask_dir" -name '*.png' | wc -l)" -eq 0 ]; then
                existing_masks_ok=false
                break
            fi
        done

        if [ "$existing_masks_ok" = true ]; then
            echo -e "${BLUE}  $series: Masks already exist, skipping segmentation${NC}"
            continue
        fi

        echo -e "${YELLOW}  $series: Missing expected masks, running segmentation${NC}"
    fi

    if [ -d "$segmentation_output" ]; then
        rm -rf "$segmentation_output"
    fi
    mkdir -p "$segmentation_base"
    
    poetry run python -m src.preprocessing.segmentation.segmentation \
        --tree "$series" \
        $MASK_ARGS 2>/dev/null || {
        echo -e "${RED}✗ $series: Segmentation failed${NC}"
        continue
    }
done
echo -e "${GREEN}✓ Segmentation complete${NC}"

echo -e "${YELLOW}[4/5] Creating Datumaro datasets...${NC}"
for series in "${series_list[@]}"; do
    segmentation_output="$segmentation_base/$series"
    if [ ! -d "$segmentation_output" ]; then
        continue
    fi
    
    final_zip="$segmentation_base/datumaro_${series}.zip"

    if [ "$SKIP_DATUMARO" = true ] && [ -f "$final_zip" ]; then
        echo -e "${BLUE}  $series: Datumaro zip already exists, skipping conversion${NC}"
        datumaro_files+=("$final_zip")
        continue
    fi

    datumaro_args=()
    if [ "$DATUMARO_NO_MEDIA" = true ]; then
        datumaro_args+=(--no-media)
        # Existing-job annotation import needs IDs matching task frame names.
        # CVAT commonly expects extensionless frame ids for Datumaro imports.
        datumaro_args+=(--item-id-mode relative_stem)
    fi
    
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    poetry run python src/preprocessing/conversion/mask2datumaro.py \
        --segmentation-output "$segmentation_output" \
        --output "$final_zip" \
        --task-name "$series" \
        "${datumaro_args[@]}" 2>/dev/null || {
        echo -e "${RED}✗ $series: Datumaro conversion failed${NC}"
        continue
    }
    
    datumaro_files+=("$final_zip")
done
echo -e "${GREEN}✓ Datumaro datasets created${NC}"

# Upload to CVAT if enabled and files exist
if [ "$UPLOAD" = true ]; then
    if [ ${#datumaro_files[@]} -eq 0 ]; then
        echo -e "${YELLOW}[5/5] No datumaro files to upload${NC}"
    elif [ -z "${CVAT_TOKEN:-}" ]; then
        echo -e "${RED}[5/5] Cannot upload: CVAT_TOKEN must be set in .env${NC}"
        exit 1
    elif [ -z "$UPLOAD_JOB_ID" ] && [ -z "${CVAT_JOB_ID:-}" ] && [ -z "${CVAT_PROJECT_ID:-}" ]; then
        echo -e "${RED}[5/5] Cannot upload: set CVAT_PROJECT_ID or provide --upload-job-id/CVAT_JOB_ID${NC}"
        exit 1
    else
        echo -e "${YELLOW}[5/5] Uploading to CVAT...${NC}"
        upload_args=(--organization "${CVAT_ORGANIZATION:-BP}")
        if [ -n "$TREE" ]; then
            upload_args+=(--file "datumaro_${TREE}.zip")
        elif [ -n "${CVAT_UPLOAD_FILE:-}" ]; then
            upload_args+=(--file "${CVAT_UPLOAD_FILE}")
        fi
        if [ -n "$UPLOAD_JOB_ID" ]; then
            upload_args+=(--job-id "$UPLOAD_JOB_ID")
        fi

        poetry run python src/preprocessing/upload_to_cvat.py "${upload_args[@]}" 2>/dev/null || {
            echo -e "${RED}✗ CVAT upload failed${NC}"
            exit 1
        }
        echo -e "${GREEN}✓ Upload complete${NC}"
    fi
else
    echo -e "${YELLOW}[5/5] Skipping CVAT upload (--upload not specified)${NC}"
fi

echo ""
echo -e "${GREEN}✓ Pipeline complete!${NC}"
echo "  Output: $segmentation_base/datumaro_*.zip"