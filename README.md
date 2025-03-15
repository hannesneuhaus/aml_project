# aml_project

## Installation Information

This is written using python 3.11. To ensure that this works properly, please ensure that the correct version is used.
Install all necessary requirements using the requirements.txt file using the following command
pip install -r ./requirements.txt

## Usage

To receive correct predictions is necessary to use the exact name of the movie as it is used in the training data. The easiest way currently is to copy
the name of the movie from the movies.csv file.
To interact with the model there are two possible ways. It is possible to give the model movies as a list (compare recommender_notebook, cell six), and then run the method.
But there is also a terminal controller that allowes the user to add movies via a simple terminal menu.
Running the scripy.py will automatically prepare the data, train the model and start the terminal menu. This will take time

### Licensing

Neither the University of Minnesota nor any of the researchers involved can guarantee the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

- The user may not state or imply any endorsement from the University of Minnesota or the GroupLens Research Group.
- The user must acknowledge the use of the data set in publications resulting from the use of the data set (see below for citation information).
- The user may redistribute the data set, including transformations, so long as it is distributed under these same license conditions.
- The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota.
- The executable software scripts are provided "as is" without warranty of any kind, either expressed or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. The entire risk as to the quality and performance of them is with you. Should the program prove defective, you assume the cost of all necessary servicing, repair or correction.

In no event shall the University of Minnesota, its affiliates or employees be liable to you for any damages arising out of the use or inability to use these programs (including but not limited to loss of data or data being rendered inaccurate).

If you have any further questions or comments, please email <grouplens-info@umn.edu>
