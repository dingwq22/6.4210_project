
## Setup using Docker

1. Install Docker
2. Install the VSCode Dev Container Extension.
3. Open your pset/project folder in VSCode
4. Create the Dockerfile in the folder by running this command in the VSCode terminal  
`echo "FROM russtedrake/manipulation:latest" > Dockerfile`  
5. Reopen your folder in the Dev Container by clicking the blue icon in the bottom left corner, and clicking “Reopen in Container.” Select “From ‘Dockerfile’”. No additional features are necessary.
6. You may have to reinstall some VSCode extensions in the container, such as the Jupyter notebook extension or Python Intellisense. However, you should be able to run all of the notebooks from class now! The terminal is now just a Linux terminal, inside of the container.