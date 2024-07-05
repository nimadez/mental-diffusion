#!/bin/bash

# for debian-based linux distributions

# install bash scripts to "/usr/local/bin"
# to call the script from wherever you are

scr=$(realpath "$0")
dir=$(dirname "$scr")

# create scripts in ~/.cache
cat > ~/.cache/mdx <<EOF
#!/bin/bash
~/.venv/mdx/bin/python3 $dir/src/mdx.py "\$@"
EOF

cat > ~/.cache/upscale <<EOF
#!/bin/bash
~/.venv/mdx/bin/python3 $dir/src/addons/upscale.py "\$@"
EOF

# copy scripts to /usr/local/bin/
sudo cp ~/.cache/mdx /usr/local/bin/
sudo chmod +x /usr/local/bin/mdx

sudo cp ~/.cache/upscale /usr/local/bin/
sudo chmod +x /usr/local/bin/upscale

# remove temp files
rm ~/.cache/mdx
rm ~/.cache/upscale

echo Done
