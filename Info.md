## **Folder Structure**
---

&nbsp;

At the location where main.py/.exe exists,

        |_____Datasets/
              |
              |_____Part_Name_1/
              |     |
              |     |_____Positive/
              |     |     |_____(Positive Samples).png
              |     |
              |     |_____Negative/
              |     |     |_____(Negative Samples).png
              |     |
              |     |_____Checkpoints/
              |     |     |_____State.pt and Metrics.txt
              |     |
              |     |_____Graphs.jpg
              |     |
              |     |_____Positive_Features.npy
              |     |
              |     |_____Negative_Features.npy
              |
              |_____Part_Name_2/
                    |
                    |_____Positive/
                    |     |_____(Positive Samples).png
                    |
                    |_____Negative/
                    |     |_____(Negative Samples).png
                    |
                    |_____Checkpoints/
                    |     |_____State.pt and Metrics.txt
                    |
                    |_____Graphs.jpg
                    |
                    |_____Positive_Features.npy
                    |
                    |_____Negative_Features.npy
              .
              .
              .
              .
              .
              .


&nbsp;

---

&nbsp;

## **Python Scripts Information**

&nbsp;


1. *cli.py* - Contains the Command Line Interface Implementation of the Application.

2. *gui.py* - Contains the Graphical User Interface Implementation of the Application.

&nbsp;

3. *DatasetTemplates.py* - Pytorch Dataset Templates used by the Application; needed so that data can be batch processed efficiently.

4. *MakeData.py* - Contains the Dataset Creation Pipeline.

&nbsp;

5. *Models.py* - Pytorch Models used by the Application; 
    - A Feature Extraction Model
    - A Region-of-Interest Detection/Extraction Model
    - A Siamese Neural Network
6. *Train.py* - Contains the fit() function; is responsible for the entire training process employed in the Application.

&nbsp;

7. *Snapshot.py* - Used only by the CLI Version of the Application; handles capture a frame from the webcam feed.
8. *RTApp.py* - Used only by the CLI Version of the Application; handles inference.

&nbsp;

9. *utils.py* - Contains Constants and Utility Functions used throughout the Application.

&nbsp;

10. *main.py* - Entry Point into the Application.

&nbsp;

---

&nbsp;

- To build an exe using pyinstaller, go into the directory containing `main.py` and run the command,

`pyinstaller main.py --onefile --clean --distpath "./" --name Application --hidden-import=skimage.filters.rank.core_cy_3d --log-level=DEBUG`

&nbsp;

---
