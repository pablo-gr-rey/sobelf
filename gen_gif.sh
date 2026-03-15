# USAGE: ./gen_gif.sh [frames] [width] [height]

source .venv/bin/activate
mandelcli --frames $1 --zxoffset -0.743643887037151 --zyoffset 0.131825904205330 --width $2 --height $3 --zoominc 2 --zoom 2 --filepath temp/
convert -delay 50 -loop 0 +remap temp/*.png anim.gif
rm temp/frame_*.png
deactivate