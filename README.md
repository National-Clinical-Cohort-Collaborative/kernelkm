# Kernel K-Means 

This repository template is setup to publish a Python package into Foundry.

By default, the repository name at the time of the bootstrap is used as the name of the conda package.
To override this, modify the ``condaPackageName`` variable in the ``gradle.properties`` file. You will have to tick
the "Show hidden files" checkbox located in the bottom left of the Authoring interface.
 
The ``build.gradle`` file configures the publish task to only run when the repository is tagged.
You can create tag in "Branches" tab in Authoring.

Each Authoring repository has associated an Artifacts repository which holds all the produced packages. 
Libraries can currently be published to Artifacts repository (default and recommended) or to the shared channel (deprecated).
In the future, libraries will always be published to Artifacts repository.
**If you want your library to be available in Code Workbooks, you will have to publish it to the shared channel.**

**If you decide to publish a library to the shared channel, it is not safe to go back to using Artifacts repository!** 

To publish your library to the shared channel, you have to add following in ``build.gradle``:

```
pythonLibrary {	
    publishChannelName = 'libs'	
}
```



# Debugging
Open the command palette with command-shit-P
run Shell Command: Install 'code' command in PATH