#!/bin/bash

: '
Script to run Epi2me pipeline on multiple folders at the same time
Epi2me does not support this feature
Gets all inputs from all the folders and creates symlinks within a random directory and uses this directory to run the workflow.

argument structure
$1 - number of cells
$2 - kit
$3 - ref genome
$4 - output directory
$4: - fastq folder locations as the rest of the arguments
'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE=$1

echo $SCRIPT_DIR

# Parse with awk and export variables
eval "$(
awk -F= -v section="" -v script_dir="$SCRIPT_DIR" '
  /^\[/ {
    section = substr($0, 2, length($0)-2)
    gsub(/[^A-Za-z0-9_]/, "_", section)
  }
  !/^[;#]/ && /^[^\[=]+=/ {
    gsub(/^[ \t]+|[ \t]+$/, "", $0)
    key = $1
    gsub(/[ \t]/, "", key)
    gsub(/^[ \t]+|[ \t]+$/, "", $2)
    gsub(/^\.\//, script_dir "/", $2)
    value = $2
    if (section != "") {
      printf "export %s_%s=\"%s\"\n", toupper(section), toupper(key), value
    } else {
      printf "export %s=\"%s\"\n", toupper(key), value
    }
  }
' "$CONFIG_FILE"
)"

function check_or_create_dir {
    if [ ! -d $1 ]; then
      mkdir $1
      echo "Directory created ${1}"
    else
      echo "Directory already exists ${1}"
    fi
}


# read_locations=("${@:4}")

declare -a read_locations
readarray -t read_locations < "$DIRECTORY_FASTQ_READ_PATHS_FILE"

echo "-------------------------------------------------------"
echo "running Epi2me script with the following configurations"
echo "-------------------------------------------------------"

echo "OUTPUT_DIRECTORY: ${DIRECTORY_OUTPUT_DIRECTORY}"
echo "FASTQ_READ_PATHS_FILE: ${DIRECTORY_FASTQ_READ_PATHS_FILE}"
echo "TEMPORARY_DIRECTORY: ${DIRECTORY_TEMPORARY_DIRECTORY}"
echo "WORKING_DIRECTORY: ${DIRECTORY_WORKING_DIRECTORY}"
echo "NEXFLOW_CONFIG: ${DIRECTORY_NEXTFLOW_CONFIG}"
echo "NUMBA_CONFIG: ${DIRECTORY_NUMBA_CONFIG}" 

echo "EXPECTED_CELLS: ${EPI2ME_EXPECTED_CELLS}"
echo "KIT: ${EPI2ME_KIT}"
echo "REF_GENOME_DIR: ${EPI2ME_REF_GENOME_DIR}"
echo "Read Files: ${read_locations}"


check_or_create_dir $DIRECTORY_OUTPUT_DIRECTORY
check_or_create_dir $DIRECTORY_TEMPORARY_DIRECTORY
check_or_create_dir $DIRECTORY_WORKING_DIRECTORY

cd $DIRECTORY_WORKING_DIRECTORY || exit 1
pwd

cp $DIRECTORY_NEXTFLOW_CONFIG $DIRECTORY_WORKING_DIRECTORY
cp $DIRECTORY_NUMBA_CONFIG $DIRECTORY_WORKING_DIRECTORY


tempdir_location=$(uuidgen | tr -d '-')

temporary_location="${DIRECTORY_TEMPORARY_DIRECTORY}/${tempdir_location}"
mkdir -p $temporary_location


echo "created temporary directory location ${temporary_location}"
echo "Iterating through the directory locations"
for full_loc in "${read_locations[@]}"; do
	if [[ "$full_loc" = ./* ]]; then
		loc="${SCRIPT_DIR}/${full_loc#./}"
	fi

	echo $loc
	if [ -f "$loc" ]; then
		echo "$loc is a file."
		ln -s "${loc}" $temporary_location
	elif [ -d "$loc" ]; then
		echo "$loc is a directory."
		ln -s "${loc}"/* $temporary_location
	else
		echo "$loc is neither a file nor a directory."
	fi
done

nextflow run epi2me-labs/wf-single-cell  \
        --expected_cells $EPI2ME_EXPECTED_CELLS  \
        --fastq $temporary_location  \
        --kit $EPI2ME_KIT  \
        --ref_genome_dir $EPI2ME_REF_GENOME_DIR  \
        -c $DIRECTORY_NUMBA_CONFIG \
	-resume \
        -profile singularity \
        --out_dir $DIRECTORY_OUTPUT_DIRECTORY
