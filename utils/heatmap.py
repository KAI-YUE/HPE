import numpy as np

def Heatmap(confidence_img):
    """
    Plot the confidence image ([0, 1]) as heatmap.
    -----------------------------------------------------------------------------
    Args,
        confidence_img:    An confidence image(h x w).
    
    Return,
        cmap,             Colormap in BGR format. [0, 255]
    """
    cmap = np.zeros((confidence_img.shape[0], confidence_img.shape[1], 3))
    index_map = (100 * confidence_img).astype(int)
    index_map = np.where(index_map<0, 0, index_map)
    index_map = np.where(index_map>100, 100, index_map)
    cmap = cmap_matrix[index_map]
    
    return cmap[:,:,::-1]


def Gaussian_heatmap(img, center, sigma=5):
    """
    Generate a Gaussian heatmap according to g(x, y) = exp(-(x^2 + y^2)/(2*s^2))
    -----------------------------------------------------------------------------
    Args,
        img:             the original RGB image.[0,255]
        center:          tuple (x ->col, y ->row), center point of the heatmap. 
        sigma:           sigma of the Gaussian function.
        cropped_size:    output size of the cropped image
    """
    
    alpha = 0.4
    margin = 20
    [x, y] = np.meshgrid(np.arange(-margin, margin+1), np.arange(-margin, margin+1))
    g = np.exp(-(x**2 + y**2)/(2*sigma**2))
        
    big_map = np.zeros((img.shape[0]+2*margin, img.shape[1]+2*margin))
    
    # Check the validity of the center
    center[0] = min(center[0], img.shape[1]-1)
    center[0] = max(0, center[0])
    center[1] = min(center[1], img.shape[0]-1)
    center[1] = max(0, center[1])
    
    start_x = int(center[0])
    start_y = int(center[1])
    end_x = int(center[0] + 2*margin + 1)
    end_y = int(center[1] + 2*margin + 1)
    
    big_map[start_y:end_y, start_x:end_x] = g
    confidence_map = big_map[margin:-margin, margin:-margin]
    
    heatmap = Heatmap(confidence_map)
    composite = alpha * heatmap + (1 - alpha) * img
    
    return dict(confidence_map=confidence_map, heatmap=heatmap, composite=composite)

# RGB
cmap_matrix = \
np.array([
[125, 3, 254],
[115, 18, 254],
[111, 25, 254],
[105, 34, 254],
[101, 40, 254],
[95, 49, 253],
[91, 56, 253],
[85, 65, 252],
[81, 71, 252],
[75, 80, 251],
[71, 86, 251],
[65, 95, 250],
[61, 100, 249],
[55, 109, 248],
[51, 115, 248],
[45, 123, 246],
[41, 128, 246],
[35, 136, 244],
[31, 142, 243],
[25, 149, 242],
[21, 154, 241],
[15, 162, 239],
[11, 167, 238],
[5, 174, 237],
[1, 178, 236],
[4, 185, 234],
[8, 189, 232],
[14, 195, 230],
[18, 199, 229],
[24, 205, 227],
[28, 209, 226],
[34, 214, 223],
[38, 217, 222],
[44, 222, 220],
[48, 225, 218],
[54, 229, 215],
[58, 232, 214],
[64, 236, 211],
[68, 238, 209],
[74, 241, 207],
[78, 243, 205],
[84, 246, 202],
[88, 247, 200],
[94, 249, 197],
[98, 250, 195],
[104, 252, 192],
[108, 253, 190],
[114, 254, 187],
[118, 254, 185],
[124, 254, 181],
[128, 254, 179],
[134, 254, 176],
[138, 254, 174],
[144, 253, 170],
[148, 252, 168],
[154, 251, 164],
[158, 250, 162],
[164, 248, 158],
[168, 246, 156],
[174, 244, 152],
[178, 242, 149],
[184, 239, 146],
[188, 237, 143],
[194, 233, 139],
[198, 230, 136],
[204, 226, 132],
[208, 223, 130],
[214, 219, 126],
[218, 215, 123],
[224, 210, 119],
[228, 207, 116],
[234, 201, 112],
[238, 197, 109],
[244, 191, 105],
[248, 187, 102],
[254, 180, 97],
[255, 176, 95],
[255, 169, 90],
[255, 164, 87],
[255, 157, 83],
[255, 152, 80],
[255, 144, 75],
[255, 139, 72],
[255, 131, 68],
[255, 126, 65],
[255, 117, 60],
[255, 112, 57],
[255, 103, 53],
[255, 97, 49],
[255, 89, 45],
[255, 83, 42],
[255, 74, 37],
[255, 68, 34],
[255, 59, 29],
[255, 53, 26],
[255, 43, 21],
[255, 37, 18],
[255, 28, 14],
[255, 21, 10],
[255, 12, 6],
[255, 6, 3]], 
dtype=np.uint8)
    
if __name__ == '__main__':
    sigma = 5
    [x, y] = np.meshgrid(np.arange(-20, 21), np.arange(-20, 21))
    z = np.exp(-(x**2+y**2)/(2*sigma**2))
    heat_map = Heatmap(z)