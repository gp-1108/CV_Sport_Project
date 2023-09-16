The dataset consists of 15 different scenes taken from sport matches. Several sports were considered.

The Images subfolder contains the source images that you should use to measure the performance of your system.

The Masks subfolder contains, for each correspinding source image:

- im<x>_bb: the bounding boxes for player detection, in the form of a text file containing a line for each bbox with the following data: [x, y, width, height, player team ID], where (x,y) are the top-left corner coordinates and width and height are the bounding box dimensions.

- im<x>_bin: the segmentation ground truth mask, with the following coding:
  -- 0: background;
  -- 1: team 1 player;
  -- 2: team 2 player;
  -- 3: field.

- im<x>_color: the same mask, with a different color coding, useful for visualization:
  -- black: background;
  -- blue: team 1 player;
  -- red: team 2 player;
  -- green: field.
  
* PLEASE NOTE * the masks can be used only for evaluating your system. They should NEVER be used as an input to the system.