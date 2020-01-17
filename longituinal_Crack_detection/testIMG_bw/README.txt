Crop command if imagemagick is installed

mogrify -crop 960x960+0+0 -quality 100 -path ./cropped *.png

Resize command (CAUTION: resizing can hurt performance depending on the type of resize used)

mogrify -adaptive-resize 224x224 -quality 100 *.png