import PIL
import sys
from PIL import Image

# mywidth = 300
mywidth=int(sys.argv[1])
filename  = str(sys.argv[2])
ext = str(sys.argv[3])
img = Image.open(filename + '.' +ext)
wpercent = (mywidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
img.save(filename+'_resized'+ '.' +ext)