# Segmentation-of-Chest-CT-Scan-Images-for-Lung-Cancer-Detection-Using-Segment-Anything-Model-SAM-
## Overview
The project evaluates the efficacy of Segment Anything Model (SAM) in segmenting a set of chest CT scan images.
SAM is a foundation model in the field of computer vision developed by the Facebook's Artificial Intelligence Research (FAIR)
SAM employs a unique two-stage methodology in
which the input image is first encoded into a high-dimensional embedding before the
embedding and input prompt are used to generate object masks. This methodology
allows SAM to perform zero shot learning on a set of data and to accept and process
complex input prompts with different objects and attributes.

Three different methods of segmentation using SAM model were applied on the images
to segment out the lung region. The segmented lung region was further used to build a
deep learning classification model that can identify different types of lung cancer.

## Data Source and Information
The dataset used in this project was obtained from a directory in Kaggle dataset
([Download here](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)) . According to
the website, the images were fetched from different sources and grouped according to
the cancer type. There is no information regarding the accuracy of the labelling of the
data with the different cancer types and no information about the professionals that
labelled the data or the tool that was used to label the data.
The images are of the .png extension unlike most medical images in dicom format. This
is to ensure that the images can fit into the deep learning model.
The dataset contains three lung cancer types which are adenocarcinoma, large cell
carcinoma and squamous cell carcinoma and a set of normal lung chest CT scan.
Altogether, there are 1000 images.
## Project Summary
1. The first step is preprocessing of the data which involves resizing, Histogram equalisation and Gausian Blurring
2. The extraction of the lung region from the preprocessed data using the three methods of segmentation with SAM model.
   - **Automatic Mask Generation**: With this method, the model segments every part of the image
     ~~~
    
          def show_anns(anns):
              if len(anns) == 0:
                  return
              sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
              ax = plt.gca()
              ax.set_autoscale_on(False)
          
              img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
              img[:,:,3] = 0
              for ann in sorted_anns:
                  m = ann['segmentation']
                  color_mask = np.concatenate([np.random.random(3), [0.35]])
                  img[m] = color_mask
              ax.imshow(img)

   ![Automatic image](https://github.com/user-attachments/assets/b30aaf4c-0e07-4e2a-a22b-1fe3a419d5a8)

   - **Prompt-based Mask Generation**:
     This is a method that involves the use of input prompts to segment specific regions of
interest in an image
     ~~~
        def show_mask(mask, ax, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
        
        def show_points(coords, labels, ax, marker_size=375):
              pos_points = coords[labels==1]
              neg_points = coords[labels==0]
              ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
              ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
            
        def show_box(box, ax):
              x0, y0 = box[0], box[1]
              w, h = box[2] - box[0], box[3] - box[1]
              ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
   ![prompt image](https://github.com/user-attachments/assets/f48404ed-7630-497f-b263-1bdad9221534)

   - **Interface-based Mask Generation**
     This is a User-Interface (UI) version of the SAM. For now, the interface is hosted on the
SAM website [here](https://segment-anything.com/demo). It allows users to upload an image into
the platform and the platform applies the SAM model in the background. Depending on
the users need, this model can apply multiple or specific masks on the image. There are
three ways of creating masks using this method.
3. The third step is the use of the extracted images to build a classification model in Convolutional Neural Networt (CNN).
## Result
Results
show that SAM model was able to segment out the lung region and performed very well
in terms of stability score and IoU in the segmentation process. Accuracy of 93% was
achieved with the CNN classifier using VGG16 model architecture. 
