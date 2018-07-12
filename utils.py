import numpy as np

def get_subwindows(im, height = 256, width = 256, pad_size = 21, top_edge=-1, right_edge=1, bottom_edge=2, left_edge=-2, middle=0):
    y_stride, x_stride, = width - (2 * pad_size), height - (2 * pad_size)
    if (height > im.shape[0]) or (width > im.shape[1]):
        print("Invalid crop: crop dims larger than image ")
        raise Exception
    ims = list()
    bin_ims = list()
    locations = list()
    y = 0
    y_done = False
    while y <= im.shape[0] and not y_done:
        x = 0
        if y + height > im.shape[0]:
            y = im.shape[0] - height
            y_done = True
        x_done = False
        while x <= im.shape[1] and not x_done:
            if x + width > im.shape[1]:
                x = im.shape[1] - width
                x_done = True
            locations.append(((y, x, y + height, x + width),
                              (y + pad_size, x + pad_size, y + y_stride, x + x_stride),
                              top_edge if y == 0 else (bottom_edge if y == (im.shape[0] - height) else middle),
                              left_edge if x == 0 else (right_edge if x == (im.shape[1] - width) else middle)
                              ))
            ims.append(im[y:y + height, x:x + width, :])
            x += x_stride
        y += y_stride

    return locations, ims

def stich_together(locations, subwindows, size):
    TILE_SIZE = 256
    PADDING_SIZE = 21
    LEFT_EDGE = -2
    TOP_EDGE = -1
    MIDDLE = 0
    RIGHT_EDGE = 1
    BOTTOM_EDGE = 2
    print("Size:", size)
    output = np.zeros(size, dtype=np.float32)
    for location, subwindow in zip(locations, subwindows):
        outer_bounding_box, inner_bounding_box, y_type, x_type = location
        y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1
        # print outer_bounding_box, inner_bounding_box, y_type, x_type

        if y_type == TOP_EDGE:
            y_cut = 0
            y_paste = 0
            height_paste = TILE_SIZE - PADDING_SIZE
        elif y_type == MIDDLE:
            y_cut = PADDING_SIZE
            y_paste = inner_bounding_box[0]
            height_paste = TILE_SIZE - 2 * PADDING_SIZE
        elif y_type == BOTTOM_EDGE:
            y_cut = PADDING_SIZE
            y_paste = inner_bounding_box[0]
            height_paste = TILE_SIZE - PADDING_SIZE

        if x_type == LEFT_EDGE:
            x_cut = 0
            x_paste = 0
            width_paste = TILE_SIZE - PADDING_SIZE
        elif x_type == MIDDLE:
            x_cut = PADDING_SIZE
            x_paste = inner_bounding_box[1]
            width_paste = TILE_SIZE - 2 * PADDING_SIZE
        elif x_type == RIGHT_EDGE:
            x_cut = PADDING_SIZE
            x_paste = inner_bounding_box[1]
            width_paste = TILE_SIZE - PADDING_SIZE

        # print (y_paste, x_paste), (height_paste, width_paste), (y_cut, x_cut)

        output[y_paste:y_paste + height_paste, x_paste:x_paste + width_paste] = subwindow[y_cut:y_cut + height_paste,
                                                                                x_cut:x_cut + width_paste]

    return output