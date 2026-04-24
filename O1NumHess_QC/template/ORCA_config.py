# modify and write your configs in it.
from textwrap import dedent


config = [
    # below is an example of config, modify it according to your condition of how to run ORCA.
    # you can easily write more than one config if you have more than one config to run the program.
    {
        "name": "ORCA", # unique name between different configurations
        "bash": dedent(
            # put your bash command below for running ORCA successfully
            """
            #!/bin/bash
            # openmpi
            MPI_HOME=/usr/local/openmpi
            export PATH=${MPI_HOME}/bin:$PATH
            export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
            export MANPATH=${MPI_HOME}/share/man:$MANPATH

            # ORCA 6.0.1 secion
            export LD_LIBRARY_PATH=/path/to/orca:$LD_LIBRARY_PATH
            export PATH=/path/to/orca:$PATH
            """
        ).lstrip(), # use lstrip() to remove the first empty line before #!/bin/bash
        "path": r"/<path to orca>/orca", # program path
    },
]
