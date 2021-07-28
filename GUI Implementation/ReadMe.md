- Contains the GUI Implementation of the Application. Copy `Siamese Only GUI\gui.py` to `Siamese*\gui.py` once design has been finalized

Fixes:
1. Error Handling breaks the application. Add try, catch.
    1. ZeroDivisionError possible from make_data.
    2. FileNotFoundError if user enters the wrong component/part name.
2. Add Comments
3. Breaks if used on screens of different sizes. (Not a problem)
    1. To get the application to work on a smaller screen, change pack method to grid. However, this will spoil the layout.
