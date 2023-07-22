import matplotlib.pyplot as plt

def show_image(image_data, class_names, classes_names, class_preds = None):
  """
  this functions takes in 9 batches of array of numbers in height, weight and color format functions
  and returns a 3 by 3 visualization of the data

  args
  ----
  image_data: (array) arrays of numbers in height, weight and color format 
  class_names: (list) class names of the data
  class_preds: (Optional) if available, The Target name that the model predicted
  smooth:(Boolean) either pixelated or smooth version of the dataset
  """

  fig= plt.figure(figsize=(9,9))
  rows, cols=3,3
  for i in range(1, rows*cols+1):
    img= image_data[i]
    label= classes_names[class_names[i]]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img,
               #interpolation= 'spline16',
               interpolation_stage='rgba')

    if class_preds is None:
      plt.title(label)
    else:
      pred= classes_names[class_preds[i]]
      title_text= f'Truth:{label} | Pred:{pred}'
      if label==pred:
        plt.title(title_text, fontsize=10, c='g') 
      else:
        plt.title(title_text, fontsize=10, c='r')
  
    plt.axis(False)
  