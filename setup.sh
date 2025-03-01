# Eric Fischer / emfischer712@ucla.edu
# www.ericmfischer.com

# extract the zipped pre-trained models and datasets
find . -name "*.zip" | while read filename; do unzip -o -d "`dirname "$filename"`" "$filename"; done;
